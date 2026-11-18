"""Running and scoring the AI-judge baselines with the frozen judges-v1 prompts (judges.py).

run: send each prompt to an OpenAI-compatible chat endpoint (a local server such as Ollama or
  LM Studio, or a paid API) and append the replies to eval/judges/answers_<judge>_<split>.jsonl.
  It will not start without a spending rule: local=True for a free local server, or prices per
  million tokens plus a max_usd cap, checked before every call. It skips runs already answered, so
  it resumes after an interruption.
score: each judge next to DoneCheck on the same runs, from the saved answers and the committed
  predictions; writes eval/results/judges_<split>.json. Prompts never hold a result (judges.py).
"""

import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from donecheck import judges
from donecheck.loader import iter_runs
from donecheck.scoring import summary

JUDGE_DIR = Path("eval/judges")
PREDICTIONS = {"building": "eval/results/building_predictions.tsv", "final": "eval/results/final_glm45_predictions.tsv"}
MAX_REPLY_TOKENS = 200


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
        usd_per_mtok_out: float | None = None, max_usd: float | None = None, send=post) -> dict:
    if not local and None in (usd_per_mtok_in, usd_per_mtok_out, max_usd):
        raise SystemExit("set local=True for a free local server, or both prices and max_usd for a paid one")
    prompts = [json.loads(line) for line in (JUDGE_DIR / split / f"prompts_{judge}.jsonl").open(encoding="utf-8")]
    out = JUDGE_DIR / f"answers_{judge}_{split}.jsonl"
    done = set(judges.read_answers(out)) if out.exists() else set()
    spent = sum(a.get("cost_usd") or 0 for a in judges.read_answers(out).values()) if out.exists() else 0.0
    sent = 0
    with out.open("a", encoding="utf-8") as fh:
        for p in prompts:
            if (p["model"], p["instance_id"]) in done:
                continue
            if not local:  # worst case for this call: about 1 token per 3 characters in, the full reply out
                worst = len(p["prompt"]) / 3 * usd_per_mtok_in / 1e6 + MAX_REPLY_TOKENS * usd_per_mtok_out / 1e6
                if spent + worst > max_usd:
                    return {"sent": sent, "spent_usd": spent, "stopped": f"the next call could pass max_usd={max_usd}"}
            text, tokens_in, tokens_out = send(endpoint, model, p["prompt"])
            cost = 0.0 if local else tokens_in * usd_per_mtok_in / 1e6 + tokens_out * usd_per_mtok_out / 1e6
            spent += cost
            sent += 1
            fh.write(json.dumps({"model": p["model"], "instance_id": p["instance_id"], "judge": judge,
                                 "prompt_version": p["prompt_version"], "judge_model": model,
                                 "endpoint": urllib.parse.urlsplit(endpoint).netloc, "answer": text,
                                 "input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": cost}) + "\n")
            fh.flush()
    return {"sent": sent, "spent_usd": spent, "stopped": None}


def score(split: str) -> dict:
    with open(PREDICTIONS[split], newline="", encoding="utf-8") as fh:
        preds = {(r["model"], r["instance_id"]): r for r in csv.DictReader(fh, delimiter="\t")}
    out = {"split": split, "prompt_version": judges.PROMPT_VERSION}
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
        out[judge] = {"runs": len(keys), "answers_not_in_predictions": len(answers) - len(keys),
                      "no_answer": sum(answers[k]["p"] is None for k in keys),
                      "judge_models": sorted({answers[k].get("judge_model", "?") for k in keys}),
                      "cost_usd_per_run": float(np.mean([answers[k].get("cost_usd") or 0 for k in keys])),
                      "judge": summary(y, judge_scores, bugs), "donecheck_same_runs": summary(y, donecheck_scores, bugs)}
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path(f"eval/results/judges_{split}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
