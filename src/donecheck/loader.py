"""Load mini-SWE-agent run files in the text shape (agent versions 1.7.0 to 1.17.x).

load_run reads one run file into a Run: the facts about the whole run, plus its steps.
iter_runs reads every run under a data folder. Why each field is read the way it is:
section 7 of the master plan.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

RUN_FORMAT = "mini-swe-agent-1"
TASK_END = "</pr_description>"
# The submit word named in the instructions tells the two instruction wordings apart.
OLDER_SUBMIT_WORD = "MINI_SWE_AGENT_FINAL_OUTPUT"
NEWER_SUBMIT_WORD = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# How the agent finds its command: exactly one bash block in its message. Agent 1.7.0
# uses the first pattern; agent 1.17 also allows spaces after "bash". The reply that
# follows, not these patterns, decides whether a command ran.
BLOCK_PATTERNS = (
    re.compile(r"```bash\n(.*?)\n```", re.DOTALL),
    re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL),
)
ANY_BLOCK = BLOCK_PATTERNS[1]

# The replies the agent framework writes back, exactly as they appear in the run files.
RAN = re.compile(r"<returncode>(-?\d+)</returncode>\n<output>\n(.*)</output>", re.DOTALL)
# Outputs over 10,000 characters keep their first and last 5,000 characters.
CUT = re.compile(
    r"<returncode>(-?\d+)</returncode>\n<warning>\n.*?\n</warning>"
    r"<output_head>\n(.*)\n</output_head>\n"
    r"<elided_chars>\n(\d+) characters elided\n</elided_chars>\n"
    r"<output_tail>\n(.*)\n</output_tail>",
    re.DOTALL,
)
TIMEOUT = re.compile(
    r"The last command <command>.*?</command> timed out and has been killed\.\n"
    r"The output of the command was:\n ?<output>\n(.*)\n</output>\n"
    r"Please try another command and make sure to avoid those requiring interactive input\.",
    re.DOTALL,
)
FORMAT_ERROR = re.compile(r"Please always provide EXACTLY ONE action in triple backticks, found \d+ actions\.")


class RunFileError(ValueError):
    """A run file or results file is not in the shape this loader expects."""


@dataclass(frozen=True)
class Step:
    message: int  # index of the agent's message in the run's messages
    note: str  # the agent's text outside its bash blocks
    outcome: str  # "ran", "timeout", "format_error" (nothing ran), "submitted" or "no_reply"
    command: str | None  # the bash block the agent ran; None for a format error
    exit_code: int | None  # only when the outcome is "ran"
    output: str  # what the command printed; for "submitted", the submission; "" for a format error
    elided_chars: int  # characters cut from the middle of a long output; 0 when nothing was cut


@dataclass(frozen=True)
class Run:
    instance_id: str
    model: str
    agent_version: str
    exit_status: str
    submitted: bool
    submission: str
    wording: str  # "older" or "newer"
    resolved: bool | None  # None when the results file has no entry for this bug
    steps: tuple[Step, ...]


def load_labels(model_dir: str | Path) -> dict[str, bool]:
    """Read a model's per_instance_details.json into {bug id: fixed or not}."""
    path = Path(model_dir) / "per_instance_details.json"
    with path.open(encoding="utf-8") as fh:
        details = json.load(fh)
    labels = {}
    for bug_id, entry in details.items():
        resolved = entry.get("resolved") if isinstance(entry, dict) else None
        if not isinstance(resolved, bool):
            raise RunFileError(f"{path}: 'resolved' for {bug_id} is {resolved!r}, not true or false")
        labels[bug_id] = resolved
    return labels


def load_run(path: str | Path, labels: dict[str, bool]) -> Run:
    """Read one run file into a Run, using the labels that load_labels gave for its model."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("trajectory_format") != RUN_FORMAT:
        raise RunFileError(f"{path}: trajectory_format is {data.get('trajectory_format')!r}, not {RUN_FORMAT!r}")
    try:
        instance_id = data["instance_id"]
        info = data["info"]
        exit_status = info["exit_status"]
        submission = info["submission"]
        agent_version = info["mini_version"]
        messages = data["messages"]
        task = messages[1]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise RunFileError(f"{path}: missing field {err}") from err
    if not isinstance(submission, str):
        raise RunFileError(f"{path}: info.submission is {type(submission).__name__}, not text")

    return Run(
        instance_id=instance_id,
        model=model_name(path),
        agent_version=agent_version,
        exit_status=exit_status,
        # From exit_status alone: a submitted run can have an empty change,
        # and a crashed run's submission holds an error message instead of a diff.
        submitted=exit_status == "Submitted",
        submission=submission,
        wording=instruction_wording(task, path),
        # .get, not [...]: a bug missing from the results file is unknown (None), not "not fixed".
        resolved=labels.get(instance_id),
        steps=parse_steps(messages, exit_status, submission, path),
    )


def iter_runs(data_dir: str | Path) -> Iterator[Run]:
    """Every run under data_dir/<model folder>/trajs/, one model folder at a time, in name order."""
    for model_dir in sorted(p for p in Path(data_dir).iterdir() if (p / "trajs").is_dir()):
        labels = load_labels(model_dir)
        for path in sorted((model_dir / "trajs").glob("*.traj.json")):
            yield load_run(path, labels)


def model_name(path: Path) -> str:
    """The model, from the folder two levels up: 20250807_mini-v1.7.0_gpt-5-nano gives gpt-5-nano."""
    folder = path.parent.parent.name
    parts = folder.split("_", 2)
    if len(parts) != 3 or not parts[1].startswith("mini-v"):
        raise RunFileError(f"{path}: folder {folder!r} is not named <date>_mini-v<version>_<model>")
    return parts[2]


def instruction_wording(task: object, path: Path) -> str:
    """Which instructions the agent got, "older" or "newer", read from the text after the bug report.

    The older wording names MINI_SWE_AGENT_FINAL_OUTPUT and also lets the agent submit when
    it can't make further progress; the newer one names COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
    """
    if not isinstance(task, str) or TASK_END not in task:
        raise RunFileError(f"{path}: the task message has no {TASK_END}")
    instructions = task.split(TASK_END, 1)[1]
    older = OLDER_SUBMIT_WORD in instructions
    newer = NEWER_SUBMIT_WORD in instructions
    if older == newer:
        raise RunFileError(f"{path}: the instructions name {'both' if older else 'neither'} of the submit words")
    return "older" if older else "newer"


def parse_steps(messages: list, exit_status: str, submission: str, path: Path) -> tuple[Step, ...]:
    """Pair each agent message after the task with the reply that follows it.

    messages[0] is the system message and messages[1] the task; after that, agent
    messages and replies alternate. A run stopped by the step or cost limit
    (LimitsExceeded) ends with one extra, empty user message.
    """
    steps = []
    last = len(messages) - 1
    for i in range(2, len(messages), 2):
        message = messages[i]
        content = message.get("content")
        if (i == last and exit_status == "LimitsExceeded" and message.get("role") == "user"
                and isinstance(content, str) and not content.strip()):
            break
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            raise RunFileError(f"{path}: messages[{i}] should be an agent message")
        reply = messages[i + 1] if i < last else None
        steps.append(parse_step(i, message["content"], reply, i + 1 == last, exit_status, submission, path))
    return tuple(steps)


def parse_step(index: int, content: str, reply: dict | None, reply_is_last: bool,
               exit_status: str, submission: str, path: Path) -> Step:
    """One agent message and what its reply says happened."""
    note = ANY_BLOCK.sub("", content).strip()
    if reply is None:  # the run ended (crashed) before the reply was written
        return Step(index, note, "no_reply", single_block(content), None, "", 0)
    text = reply.get("content")
    if reply.get("role") != "user" or not isinstance(text, str):
        raise RunFileError(f"{path}: messages[{index + 1}] should be the reply to messages[{index}]")
    if reply_is_last and exit_status == "Submitted":
        # The submit command's output, minus its first line, is the submission and ends the run.
        if text != submission:
            raise RunFileError(f"{path}: the last message is not the same as info.submission")
        return Step(index, note, "submitted", command_of(content, index, path), None, text, 0)
    if m := RAN.fullmatch(text):
        return Step(index, note, "ran", command_of(content, index, path), int(m[1]), m[2], 0)
    if m := CUT.fullmatch(text):
        output = f"{m[2]}\n[... {m[3]} characters elided ...]\n{m[4]}"
        return Step(index, note, "ran", command_of(content, index, path), int(m[1]), output, int(m[3]))
    if m := TIMEOUT.fullmatch(text):
        return Step(index, note, "timeout", command_of(content, index, path), None, m[1], 0)
    if FORMAT_ERROR.match(text):
        return Step(index, note, "format_error", None, None, "", 0)
    raise RunFileError(f"{path}: messages[{index + 1}] is not a reply this loader knows")


def single_block(content: str) -> str | None:
    """The message's one bash block, stripped as the agent strips it; None unless there is exactly one."""
    for pattern in BLOCK_PATTERNS:
        blocks = pattern.findall(content)
        if len(blocks) == 1:
            return blocks[0].strip()
    return None


def command_of(content: str, index: int, path: Path) -> str:
    """The command of a message whose reply shows that it ran."""
    command = single_block(content)
    if command is None:
        raise RunFileError(f"{path}: messages[{index}] ran a command but has no single bash block")
    return command
