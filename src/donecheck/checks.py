"""Evidence checks: facts about one run, read from its record with plain rules (plan section 8).

check_run(run) gives one run's facts as a flat dict, ready to become features on Day 4.
How a step's command is read: split_command cuts it into simple commands (it knows quotes,
$(...) and heredocs); each simple command is then taken as an edit (edits_of), a check that
runs tests or a script (check_of), or neither. result_of reads a check's result from the
step's output. Paths are read relative to /testbed, where every step starts.
"""

import fnmatch
import posixpath
import re
import shlex
from dataclasses import dataclass, field

from donecheck.loader import Run, Step

REPO = "/testbed"


@dataclass
class Part:
    """One simple command inside a step's command."""

    text: str  # the command line, without any heredoc body
    body: str = ""  # the heredoc body fed to it, if any
    words: list[str] = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.words = shlex.split(self.text)
        except ValueError:  # unbalanced quotes
            self.words = self.text.split()


HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\2")


def split_command(command: str) -> list[Part]:
    """Cut a bash command into simple commands at &&, ||, |, ; and line ends that sit outside
    quotes, $(...) and heredoc bodies. Each heredoc body goes to the command that opened it."""
    parts: list[Part] = []
    buf: list[str] = []
    pending: list[tuple[str, bool, int]] = []  # heredocs still to read: delimiter, tabs stripped, owner
    quote, depth, i, n = "", 0, 0, len(command)

    def flush() -> None:
        text = "".join(buf).strip()
        buf.clear()
        if text:
            parts.append(Part(text))

    def read_bodies(i: int) -> int:
        for delimiter, strip_tabs, owner in pending:
            lines = []
            while i < n:
                end = command.find("\n", i)
                end = n if end == -1 else end
                line, i = command[i:end], end + 1
                if (line.lstrip("\t") if strip_tabs else line).rstrip() == delimiter:
                    break
                lines.append(line)
            if owner < len(parts):
                parts[owner].body = "\n".join(lines)
        pending.clear()
        return i

    while i < n:
        c = command[i]
        if quote:
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i:i + 2])
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "\\" and i + 1 < n:
            if command[i + 1] != "\n":  # a backslash at a line end joins two lines
                buf.append(command[i:i + 2])
            i += 2
            continue
        elif command.startswith("$(", i):
            depth += 1
        elif c == ")" and depth:
            depth -= 1
        elif not depth:
            if c == "<" and (m := HEREDOC.match(command, i)) and not command.startswith("<<<", i):
                pending.append((m[3], m[1] == "-", len(parts)))
                buf.append(m[0])
                i = m.end()
                continue
            if c in "\n;|" or command.startswith("&&", i):
                flush()
                i += 2 if command.startswith(("&&", "||"), i) else 1
                if c == "\n" and pending:
                    i = read_bodies(i)
                continue
        buf.append(c)
        i += 1
    flush()
    return parts


WRAPPERS = {"timeout", "env", "time", "nohup", "sudo", "stdbuf", "nice"}
SETTING = re.compile(r"[A-Za-z_]\w*=")
REDIRECT = re.compile(r"(?:\d|&)?>>?(.*)")


def program(words: list[str]) -> list[str]:
    """The words from the program on, without leading VAR=value settings and wrappers like timeout 60."""
    i = 0
    while i < len(words):
        if SETTING.match(words[i]):
            i += 1
        elif words[i] in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or words[i][:1].isdigit()):
                i += 1
        else:
            break
    return words[i:]


def split_redirects(words: list[str]) -> tuple[list[str], list[str]]:
    """(the arguments, the files that output is written into) of a command's words."""
    args, targets, skip = [], [], False
    for i, word in enumerate(words):
        if skip:
            skip = False
        elif m := REDIRECT.fullmatch(word):
            target = m[1]
            if not target and i + 1 < len(words):
                target, skip = words[i + 1], True
            if target and not target.startswith(("&", "/dev/")):
                targets.append(target)
        elif word.startswith("<"):  # input: < file, <<EOF, <<< text
            skip = word in ("<", "<<<")
        else:
            args.append(word)
    return args, targets


def absolute(path: str, cwd: str) -> str:
    return posixpath.normpath(posixpath.join(cwd, path))


def resolve(path: str, cwd: str) -> str:
    """A path as the final diff names it: relative to /testbed, or absolute when outside it."""
    full = absolute(path, cwd)
    return full[len(REPO) + 1:] if full.startswith(REPO + "/") else "." if full == REPO else full


PYTHON = re.compile(r"(?:.*/)?python[\d.]*")
IN_PLACE = re.compile(r"-[a-zA-Z]*i.*|--in-place.*")
QUOTED = r"""[rbuf]?(['"])([^'"]+)\1"""
OPEN_WRITE = re.compile(rf"""\bopen\(\s*{QUOTED}\s*,\s*(?:mode\s*=\s*)?['"][rbt]*[wax+]""")
PATH_WRITE = re.compile(rf"""Path\(\s*{QUOTED}\s*\)\s*\.write_(?:text|bytes)\(""")
PY_REMOVE = re.compile(rf"""\bos\.(?:remove|unlink)\(\s*{QUOTED}""")
ANY_WRITE = re.compile(r"""\.write_(?:text|bytes)\(|\bopen\((?:[^()]|\([^()]*\))*?,\s*(?:mode\s*=\s*)?['"][rbt]*[wax+]"""
                       r"""|\bos\.(?:remove|unlink|rename|replace)\(|\bshutil\.(?:copy\w*|move)\(""")
VAR_WRITE = re.compile(r"""\b\w+(?:\.\w+)*\.write_(?:text|bytes)\(|"""
                       r"""\bopen\(\s*[A-Za-z_][\w.]*(?:\[[^\]]*\])?\s*,\s*(?:mode\s*=\s*)?['"][rbt]*[wax+]""")
PATH_LITERAL = re.compile(r"""[rbuf]?(['"])((?:[\w.{}-]+/)*[\w{}-][\w.{}-]*\.(?:py|pyi|pyx|txt|cfg|toml|ini|rst|md|json|ya?ml|html|js|css|c|h))\1""")
PATCH_FILE = re.compile(r"^(?:\+\+\+ b/|\*\*\* (?:Update|Add|Delete) File: )(\S+)", re.M)
STDLIB_HELPERS = {"os", "sys", "re", "pathlib", "io", "json", "textwrap", "ast", "inspect", "glob", "shutil",
                  "difflib", "tokenize", "itertools", "collections", "fnmatch", "linecache", "string", "pprint"}
IMPORTS = re.compile(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", re.M)
READS = re.compile(r"\bopen\(|\.read_text\(|\.readlines\(")


def edits_of(part: Part, cwd: str, scripts: dict[str, str]) -> list[tuple[str, str]]:
    """The files this simple command changes, as (path as written, kind), with kind "write",
    "delete" or "revert". The path is "?" when a change is certain but its file can't be read off.
    scripts maps the files the agent wrote from heredocs to their text, so that running such a
    file counts as the edits its code makes."""
    words = program(part.words)
    if not words:
        return []
    args, targets = split_redirects(words[1:])
    found = [(t, "write") for t in targets]
    name = posixpath.basename(words[0])
    files = [a for a in args if not a.startswith("-")]
    if name in ("sed", "perl") and any(IN_PLACE.fullmatch(a) for a in args if a.startswith("-")):
        found += [(f, "write") for f in script_files(name, args)]
    elif name in ("tee", "touch", "ed", "ex"):
        found += [(f, "write") for f in files]
    elif name in ("rm", "unlink"):
        found += [(f, "delete") for f in files]
    elif name in ("mv", "cp") and len(files) >= 2:
        found.append((files[-1], "write"))
        if name == "mv":
            found += [(f, "delete") for f in files[:-1]]
    elif name == "git" and args:
        found += git_edits(args, part.body)
    elif name in ("patch", "apply_patch", "applypatch"):
        found += [(f, "write") for f in PATCH_FILE.findall(part.body)] or [("?", "write")]
    elif PYTHON.fullmatch(words[0]):
        found += python_writes(python_code(args, part.body, cwd, scripts))
    return found


def script_files(name: str, args: list[str]) -> list[str]:
    """The files a sed or perl call works on: what is left after the options and the script."""
    files, script_given, skip = [], False, False
    for a in args:
        if skip:
            skip = False
        elif a in ("-e", "--expression", "-f", "--file") or (name == "perl" and re.fullmatch(r"-\w*e", a)):
            script_given = skip = True
        elif not a.startswith("-"):
            files.append(a)
    return files if script_given else files[1:]


def git_edits(args: list[str], body: str) -> list[tuple[str, str]]:
    sub, rest = args[0], args[1:]
    if sub in ("checkout", "restore"):
        paths = rest[rest.index("--") + 1:] if "--" in rest else [p for p in rest if not p.startswith("-")
                                                                    and ("/" in p or "." in p)]
        return [("?" if p == "." else p, "revert") for p in paths]
    if sub == "stash" and rest[:1] not in (["list"], ["show"]):
        return [("?", "revert" if rest[:1] in ([], ["push"], ["save"]) else "write")]
    if sub == "reset" and "--hard" in rest:
        return [("?", "revert")]
    if sub in ("apply", "am"):
        return [(f, "write") for f in PATCH_FILE.findall(body)] or [("?", "write")]
    return []


def python_code(args: list[str], body: str, cwd: str, scripts: dict[str, str]) -> str:
    """The code a python command runs, when the record holds it."""
    if "-c" in args:
        i = args.index("-c")
        return args[i + 1] if i + 1 < len(args) else ""
    if args[:1] == ["-m"]:
        return ""
    if body and (not args or args[0] == "-"):
        return body
    script = next((a for a in args if not a.startswith("-")), "")
    return scripts.get(resolve(script, cwd), "") if script else ""


def python_writes(code: str) -> list[tuple[str, str]]:
    if not code or not ANY_WRITE.search(code):
        return []
    found = [(m[2], "write") for m in OPEN_WRITE.finditer(code)] + [(m[2], "write") for m in PATH_WRITE.finditer(code)]
    found += [(m[2], "delete") for m in PY_REMOVE.finditer(code)]
    if VAR_WRITE.search(code):  # a write whose path sits in a variable: take the file paths the code names
        found += [(m[2], "write") for m in PATH_LITERAL.finditer(code)]
    # f-string parts such as {app} become globs, and repeats go
    found = list(dict.fromkeys((re.sub(r"\{[^}]*\}", "*", p), kind) for p, kind in found))
    return found or [("?", "write")]


def only_reads(code: str) -> bool:
    """Python that reads files to print them and imports nothing but helpers: a look, not a check."""
    modules = {a or b for a, b in IMPORTS.findall(code)}
    return bool(READS.search(code)) and modules <= STDLIB_HELPERS


def check_of(part: Part, cwd: str) -> tuple[str, str] | None:
    """("test", key) when the simple command runs the repository's tests, ("script", key) when it
    runs a Python script or snippet, None otherwise. The key names what ran, to find reruns."""
    words = program(part.words)
    if not words:
        return None
    args, _ = split_redirects(words[1:])
    name = posixpath.basename(words[0])
    targets = [a for a in args if not a.startswith("-")]
    if name in ("pytest", "py.test", "tox", "nox") or words[0].endswith(("runtests.py", "bin/test")):
        return "test", " ".join([name] + targets)
    if name == "make" and any(t.startswith("test") for t in targets):
        return "test", " ".join(["make"] + targets)
    if not PYTHON.fullmatch(words[0]):
        return None
    if args[:2] in (["-m", "pytest"], ["-m", "py.test"], ["-m", "unittest"]):
        return "test", " ".join(targets)
    if args[:1] == ["-m"]:  # pip, venv and the like
        return None
    if "-c" in args or (part.body and (not args or args[0] == "-")):
        code = part.body if "-c" not in args else args[args.index("-c") + 1] if args[-1] != "-c" else ""
        return None if only_reads(code) else ("script", code)
    if not targets:  # python --version
        return None
    if targets[0].endswith(("runtests.py", "bin/test")):
        return "test", " ".join(targets)
    return "script", resolve(targets[0], cwd)


PYTEST_SUMMARY = re.compile(r"^.*\b(?:\d+ (?:passed|failed|errors?)|no tests ran)\b.* in \d+(?:\.\d+)?s\b.*$", re.M)
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|error)")
UNITTEST = re.compile(r"^Ran (\d+) tests? in .*\n+\s*(OK|FAILED)", re.M)  # unittest and Django's runtests.py
SYMPY = re.compile(r"tests finished: \d+ passed(.*?)in [\d.]+ seconds")
SYMPY_FAILED = re.compile(r"\d+ (?:failed|exceptions?)\b")
UNRELATED = re.compile(r"No module named|ImportError while (?:loading conftest|importing test module)|"
                       r"ERROR: (?:file or directory not found|not found)|command not found|can't open file|"
                       r"errors? during collection")


def result_of(step: Step) -> str:
    """What a check step's output says: "passed", "failed", "error" (it could not run what it
    meant to, such as a missing module or file), "timeout" or "unknown". Test runners' summaries
    come first, since a pipe such as | tail makes the exit code the last command's."""
    if step.outcome == "timeout":
        return "timeout"
    out = step.output
    if summaries := PYTEST_SUMMARY.findall(out):
        counts = {kind: int(n) for n, kind in PYTEST_COUNT.findall(summaries[-1])}
        if counts.get("failed") or counts.get("error"):
            return "failed"
        return "passed" if counts.get("passed") else "error"  # no tests ran
    if runs := UNITTEST.findall(out):
        ran, verdict = runs[-1]
        return "error" if ran == "0" else "passed" if verdict == "OK" else "failed"
    if runs := SYMPY.findall(out):
        return "failed" if SYMPY_FAILED.search(runs[-1]) else "passed"
    if UNRELATED.search(out):
        return "error"
    if "Traceback (most recent call last)" in out or step.exit_code not in (0, None):
        return "failed"
    return "passed" if step.exit_code == 0 else "unknown"


DIFF_FILE = re.compile(r"diff --git a/\S+ b/(\S+)")
TEST_PATH = re.compile(r"(?:^|/)(?:tests?|testing)/|(?:^|/)(?:test_[^/]*|[^/]*_tests?\.py|conftest\.py)$")
SCRATCH_NAME = re.compile(r"(?:repro|reproduce|debug|scratch|tmp|temp|check|verify|"
                          r"test_(?:repro|reproduce|fix|issue|bug|debug))", re.I)


def diff_files(submission: str) -> dict[str, str]:
    """{path: "new", "deleted" or "modified"} for each file in a submitted diff."""
    files = {}
    for block in re.split(r"(?m)^(?=diff --git )", submission):
        if m := DIFF_FILE.match(block):
            head = block[:500]
            files[m[1]] = "new" if "\nnew file mode" in head else "deleted" if "\ndeleted file mode" in head else "modified"
    return files


BACKUP = re.compile(r"\.(?:bak|orig|rej|backup|old|tmp)\d*$|~$")


def is_scratch(path: str, status: str = "new") -> bool:
    """A file the agent made for itself (a reproduction or debug script, a backup copy), not part
    of the fix: a new file at the top of the repository, or with a scratch-like name."""
    name = posixpath.basename(path)
    return status == "new" and ("/" not in path or bool(SCRATCH_NAME.match(name) or BACKUP.search(name)))


@dataclass(frozen=True)
class Event:
    step: int  # index into run.steps
    part: int  # index of the simple command within the step's command
    kind: str  # "edit", "test" or "script"
    key: str = ""  # for a check: what ran
    edits: tuple[tuple[str, str], ...] = ()  # for an edit: (path, kind) pairs
    result: str = ""  # for a check: result_of(step)

    @property
    def at(self) -> tuple[int, int]:
        return self.step, self.part


CLAIM = re.compile(r"\btests? (?:\w+ )?(?:now )?(?:pass(?:es|ed)?|succeed(?:s|ed)?)\b|"
                   r"\btests? (?:are|were|is) (?:all )?passing\b|\b(?:verified|confirmed)\b", re.I)
PLAN = re.compile(r"\b(?:let me|let's|let us|i'll|i will|need to|make sure|ensure|should|would|will|"
                  r"going to|try|if|can|could|must|might|may)\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
GAVE_UP = re.compile(r"\b(?:could ?n[o']t|can ?n[o']t|cannot|unable to|not able to|wasn't able)\b[^.\n]*?"
                     r"\b(?:fix|reproduce|resolve|run|complete|finish|install|test)|"
                     r"\bno further progress\b|\bgiving up\b", re.I)


def read_events(run: Run, is_code) -> tuple[list[Event], list[tuple[int, str]]]:
    """Every edit and check in the run, in order, plus (step, text) of every simple command."""
    events, texts, scripts = [], [], {}
    for s, step in enumerate(run.steps):
        if step.outcome not in ("ran", "timeout") or not step.command:
            continue
        cwd = REPO  # each step runs in a new shell that starts in /testbed
        for p, part in enumerate(split_command(step.command)):
            words = program(part.words)
            if not words:
                continue
            if words[0] == "cd":
                cwd = absolute(words[1], cwd) if len(words) > 1 else REPO
                continue
            if f"{posixpath.basename(words[0])}: command not found" in step.output:
                continue  # it never ran
            texts.append((s, part.text))
            edits = [(t if t == "?" else resolve(t, cwd), kind) for t, kind in edits_of(part, cwd, scripts)]
            if edits:
                events.append(Event(s, p, "edit", edits=tuple(edits)))
                if part.body:
                    scripts.update((t, part.body) for t, kind in edits if kind == "write")
            if not any(is_code(t, kind) for t, kind in edits) and (check := check_of(part, cwd)):
                events.append(Event(s, p, check[0], key=check[1], result=result_of(step)))
    return events, texts


def claim_labels(run: Run, code_edits: list[Event], checks: list[Event]) -> list[tuple[int, str]]:
    """(step, label) for each claim in the notes after the first code edit (docs/claims.md, section 3):
    supported when the last check since the last edit before the claim passed, contradicted when
    it failed or when no check ran at all, unsupported otherwise."""
    labels = []
    if not code_edits:
        return labels
    for s, step in enumerate(run.steps):
        if s <= code_edits[0].step:  # a note comes before its own command; earlier notes are findings
            continue
        claims = [x for x in SENTENCE.split(step.note) if CLAIM.search(x) and not PLAN.search(x)]
        if not claims:
            continue
        prior = [c for c in checks if c.step < s]
        last_edit = [e for e in code_edits if e.step < s][-1]
        fresh = [c for c in prior if c.at > last_edit.at]
        if not prior:
            label = "contradicted"
        elif not fresh:
            label = "unsupported"
        else:
            label = {"passed": "supported", "failed": "contradicted"}.get(fresh[-1].result, "unsupported")
        labels += [(s, label)] * len(claims)
    return labels


def check_run(run: Run) -> dict:
    """The evidence checks for one run. When no code edit is found, every check counts as after it."""
    final = diff_files(run.submission) if run.submitted else {}

    def is_code(path: str, kind: str) -> bool:
        if path == "?":  # a change whose file is unknown might be code, so it counts
            return True
        if path.startswith("/"):  # outside the repository
            return False
        if any(c in path for c in "*?["):  # a glob: code when it matches a file of the final change
            return any(fnmatch.fnmatch(f, path) and not is_scratch(f, s) for f, s in final.items())
        if path in final:
            return not is_scratch(path, final[path])
        return kind == "revert" or not is_scratch(path)

    events, texts = read_events(run, is_code)
    code_edits = [e for e in events if e.kind == "edit" and any(is_code(t, k) for t, k in e.edits)]
    checks = [e for e in events if e.kind != "edit"]
    tests = [c for c in checks if c.kind == "test"]
    last = code_edits[-1].at if code_edits else (-1, -1)
    after = [c for c in checks if c.at > last]
    passed_after = {c.key for c in after if c.result == "passed"}
    first_edit: dict[str, int] = {}
    for e in code_edits:
        for t, _ in e.edits:
            first_edit.setdefault(t, e.step)
    modified = [p for p, status in final.items() if status == "modified"]
    labelled = claim_labels(run, code_edits, checks)
    claims = [label for _, label in labelled]
    # Claims the agent stands behind at submission: made after its last code edit (docs/claims.md, section 4).
    final_claims = [label for s, label in labelled if s > code_edits[-1].step] if code_edits else []
    submit_note = run.steps[-1].note if run.submitted and run.steps else ""
    return {
        "steps": len(run.steps),
        "cost": run.cost,
        "api_calls": run.api_calls,
        "format_errors": sum(s.outcome == "format_error" for s in run.steps),
        "timeouts": sum(s.outcome == "timeout" for s in run.steps),
        "submission_empty": run.submitted and not final,
        "diff_files": len(final),
        "diff_test_files": sum(bool(TEST_PATH.search(p)) and not is_scratch(p, st) for p, st in final.items()),
        "diff_scratch_files": sum(is_scratch(p, st) for p, st in final.items()),
        "code_edits": len(code_edits),
        "unknown_edits": sum(t == "?" for e in code_edits for t, _ in e.edits),
        "checks": len(checks),
        "tests": len(tests),
        "checks_after_last_edit": len(after),
        "tests_after_last_edit": sum(c.kind == "test" for c in after),
        "passed_after_last_edit": bool(passed_after),
        "last_check_result": after[-1].result if after else "none",
        "last_test_result": tests[-1].result if tests else "none",
        "fail_then_pass": any(c.result == "failed" and c.key in passed_after for c in checks if c.at < last),
        "edited_files": len(modified),
        "edited_unopened": sum(p in first_edit and not opened_before(p, first_edit[p], texts) for p in modified),
        "claims": len(claims),
        "claims_supported": claims.count("supported"),
        "claims_unsupported": claims.count("unsupported"),
        "claims_contradicted": claims.count("contradicted"),
        "final_claims_contradicted": final_claims.count("contradicted"),
        "gave_up": bool(GAVE_UP.search(submit_note)),
    }


def opened_before(path: str, step: int, texts: list[tuple[int, str]]) -> bool:
    """Whether a command in an earlier step names the file, so its output could show it."""
    name = re.compile(rf"(?<![\w.-]){re.escape(posixpath.basename(path))}(?![\w-])")
    return any(s < step and name.search(text) for s, text in texts)
