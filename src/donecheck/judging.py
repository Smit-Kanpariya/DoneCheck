"""Running and scoring the AI-judge baselines with the frozen judges-v1 prompts (judges.py).

run: send each prompt to an OpenAI-compatible chat endpoint (a local server such as Ollama or
  LM Studio, or a paid API) and append the replies to eval/judges/answers_<judge>_<split>.jsonl.
  It will not start without a spending rule: local=True for a free local server, or prices per
  million tokens plus a max_usd cap, checked before every call. It skips runs already answered, so
  it resumes after an interruption. With work_minutes and rest_minutes it works in chunks, pausing
  after each stretch of work so a laptop running a local model can cool down.
score: each judge next to DoneCheck on the same runs, from the saved answers and the committed
  predictions; writes eval/results/judges_<split>.json. On the building sample it also finds each
  judge's threshold: the lowest that flags at most 5% of fixed runs, the rule DoneCheck's threshold
  follows. On GLM-4.5 each judge is read at that threshold, and DoneCheck at its frozen one.
freeze: before any GLM-4.5 call, record the judge model as the local server describes it (digest,
  settings, runtime version) and the building thresholds in eval/judge_freeze.json. run refuses the
  final split until that file exists and names the same model. Prompts never hold a result (judges.py).
"""

import csv
import http.client
import json
import math
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from donecheck import judges
from donecheck.loader import iter_runs
from donecheck.scoring import summary, threshold_at

JUDGE_DIR = Path("eval/judges")
PREDICTIONS = {"building": "eval/results/building_predictions.tsv", "final": "eval/results/final_glm45_predictions.tsv"}
MAX_REPLY_TOKENS = 200
FREEZE = Path("eval/judge_freeze.json")
THRESHOLD_RULE = "the lowest threshold that flags at most 5% of fixed runs in the 200-run building sample"


def write_final_prompts(data_dir: str = "data_final") -> None:
    """The frozen prompts for every submitted GLM-4.5 run."""
    judges.write_prompts([r for r in iter_runs(data_dir) if r.submitted], JUDGE_DIR / "final")


def post(endpoint: str, model: str, prompt: str) -> tuple[str, int, int]:
    """One chat call; returns the reply and the input and output token counts the server reports."""
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("JUDGE_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": MAX_REPLY_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(request, timeout=600) as r:
        reply = json.loads(r.read())
    usage = reply.get("usage") or {}
    return reply["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def run(split: str, judge: str, endpoint: str, model: str, local: bool = False, usd_per_mtok_in: float | None = None,
        usd_per_mtok_out: float | None = None, max_usd: float | None = None, send=post,
        work_minutes: float | None = None, rest_minutes: float = 0) -> dict:
    if not local and None in (usd_per_mtok_in, usd_per_mtok_out, max_usd):
        raise SystemExit("set local=True for a free local server, or both prices and max_usd for a paid one")
    if split == "final":
        frozen = json.loads(FREEZE.read_text()) if FREEZE.exists() else {}
        if frozen.get("judge_model") != model:
            raise SystemExit(f"freeze the judge setup on the building sample first ({FREEZE} must name {model!r})")
    prompts = [json.loads(line) for line in (JUDGE_DIR / split / f"prompts_{judge}.jsonl").open(encoding="utf-8")]
    out = JUDGE_DIR / f"answers_{judge}_{split}.jsonl"
    done = set(judges.read_answers(out)) if out.exists() else set()
    spent = sum(a.get("cost_usd") or 0 for a in judges.read_answers(out).values()) if out.exists() else 0.0
    sent, chunk_started = 0, time.monotonic()
    with out.open("a", encoding="utf-8") as fh:
        for p in prompts:
            if (p["model"], p["instance_id"]) in done:
                continue
            if work_minutes and time.monotonic() - chunk_started >= 60 * work_minutes:
                time.sleep(60 * rest_minutes)  # cool down between chunks
                chunk_started = time.monotonic()
            if not local:  # worst case for this call: about 1 token per 3 characters in, the full reply out
                worst = len(p["prompt"]) / 3 * usd_per_mtok_in / 1e6 + MAX_REPLY_TOKENS * usd_per_mtok_out / 1e6
                if spent + worst > max_usd:
                    return {"sent": sent, "spent_usd": spent, "stopped": f"the next call could pass max_usd={max_usd}"}
            for tries in range(3):  # a server restart or a slow reply; then stop, and a rerun resumes here
                try:
                    started = time.monotonic()
                    text, tokens_in, tokens_out = send(endpoint, model, p["prompt"])
                    break
                except (OSError, http.client.HTTPException, KeyError, ValueError) as err:
                    if tries == 2:
                        raise SystemExit(f"{p['model']}/{p['instance_id']}: {err!r}; rerun to resume") from err
                    time.sleep(10 * (tries + 1))
            seconds = time.monotonic() - started
            cost = 0.0 if local else tokens_in * usd_per_mtok_in / 1e6 + tokens_out * usd_per_mtok_out / 1e6
            spent += cost
            sent += 1
            fh.write(json.dumps({"model": p["model"], "instance_id": p["instance_id"], "judge": judge,
                                 "prompt_version": p["prompt_version"], "judge_model": model,
                                 "endpoint": urllib.parse.urlsplit(endpoint).netloc, "answer": text,
                                 "input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": cost,
                                 "seconds": round(seconds, 2)}) + "\n")
            fh.flush()
    return {"sent": sent, "spent_usd": spent, "stopped": None}


def score(split: str) -> dict:
    with open(PREDICTIONS[split], newline="", encoding="utf-8") as fh:
        preds = {(r["model"], r["instance_id"]): r for r in csv.DictReader(fh, delimiter="\t")}
    out = {"split": split, "prompt_version": judges.PROMPT_VERSION}
    frozen = json.loads(FREEZE.read_text()) if split == "final" else None
    donecheck_threshold = json.loads(Path("models/donecheck-v1.json").read_text())["threshold"] if frozen else None
    if frozen:
        out["judge_freeze"] = {k: frozen[k] for k in ("judge_model", "digest", "runtime", "frozen_at")}
    for judge in judges.JUDGES:
        path = JUDGE_DIR / f"answers_{judge}_{split}.jsonl"
        if not path.exists():
            out[judge] = "pending: no answers yet"
            continue
        answers = judges.read_answers(path)
        keys = [k for k in answers if k in preds]
        y = np.array([preds[k]["false_done"] == "1" for k in keys])
        bugs = np.array([k[1] for k in keys])
        judge_scores = np.array([0.5 if answers[k]["p"] is None else answers[k]["p"] for k in keys])
        donecheck_scores = np.array([float(preds[k]["score"]) for k in keys])
        threshold = None
        if frozen:
            threshold = frozen["thresholds"][judge]
            threshold = math.inf if threshold is None else threshold  # None: no threshold kept 5%; flag nothing
        seconds = [answers[k]["seconds"] for k in keys if answers[k].get("seconds") is not None]
        out[judge] = {"runs": len(keys), "answers_not_in_predictions": len(answers) - len(keys),
                      "no_answer": sum(answers[k]["p"] is None for k in keys),
                      "judge_models": sorted({answers[k].get("judge_model", "?") for k in keys}),
                      "cost_usd_per_run": float(np.mean([answers[k].get("cost_usd") or 0 for k in keys])),
                      "seconds_per_run": float(np.mean(seconds)) if seconds else None,
                      "max_input_tokens": max(answers[k].get("input_tokens") or 0 for k in keys),
                      "judge": finite(summary(y, judge_scores, bugs, threshold)),
                      "donecheck_same_runs": summary(y, donecheck_scores, bugs, donecheck_threshold)}
        if not frozen:
            t = threshold_at(y, judge_scores)
            out[judge]["threshold_at_5"] = None if math.isinf(t) else t
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path(f"eval/results/judges_{split}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def finite(result: dict) -> dict:
    """JSON has no infinity: a threshold that flags nothing is written as None."""
    if result.get("threshold") == math.inf:
        result["threshold"] = None
    return result


def local_api(endpoint: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(endpoint.rstrip("/") + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.loads(r.read())


def freeze(model: str, endpoint: str = "http://localhost:11434") -> dict:
    """Record the local judge (as its Ollama server describes it) and the building thresholds."""
    if any((JUDGE_DIR / f"answers_{j}_final.jsonl").exists() for j in judges.JUDGES):
        raise SystemExit("GLM-4.5 answers already exist, so the judge setup can no longer be frozen")
    building = json.loads(Path("eval/results/judges_building.json").read_text())
    for j in judges.JUDGES:
        if not isinstance(building.get(j), dict) or building[j]["judge_models"] != [model]:
            raise SystemExit(f"score the building answers of {model!r} for judge {j!r} first")
        if building[j]["no_answer"] > 0.05 * building[j]["runs"]:
            raise SystemExit(f"judge {j!r} gave no usable answer on {building[j]['no_answer']} of {building[j]['runs']} "
                             "building runs; fix the setup on building data before GLM-4.5")
    tags = {m["name"]: m for m in local_api(endpoint, "/api/tags")["models"]}
    tag = tags.get(model) or tags[f"{model}:latest"]
    show = local_api(endpoint, "/api/show", {"model": model})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prompt_version": judges.PROMPT_VERSION,
           "judge_model": model, "digest": tag["digest"], "size_bytes": tag.get("size"),
           "runtime": f"ollama {local_api(endpoint, '/api/version')['version']}", "details": show.get("details"),
           "parameters": show.get("parameters"), "request": {"temperature": 0, "max_tokens": MAX_REPLY_TOKENS},
           "threshold_rule": THRESHOLD_RULE, "thresholds": {j: building[j]["threshold_at_5"] for j in judges.JUDGES},
           "building": {j: {k: building[j][k] for k in ("runs", "no_answer", "max_input_tokens")} for j in judges.JUDGES},
           "commit": commit}
    FREEZE.write_text(json.dumps(out, indent=1) + "\n")
    return out
