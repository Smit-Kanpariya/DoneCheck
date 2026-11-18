"""Write demo/index.html: the final-test numbers and real GLM-4.5 runs DoneCheck caught, missed and
falsely flagged. `python -m donecheck.demo` (needs data_final/ and eval/results/).

Examples are picked by fixed rules, not by hand (see pick). Each quotes at most 240 characters of the
agent's last note and links to the full run file on the public bucket, which is not copied here.
"""

import html
import json
from pathlib import Path

from donecheck.analysis import GROUPS, read_predictions
from donecheck.checks import check_run
from donecheck.cli import explain
from donecheck.fetch import ENTRIES, S3
from donecheck.loader import iter_runs

OUT = Path("demo/index.html")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pick(cases: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    """Per group, by fixed rules: caught with and without an empty change, the lowest-scoring miss and
    a miss whose repository tests ran after the last edit, every false alarm (up to 3), and the
    lowest-scoring accepted fix. Rules that match nothing are skipped; no run is shown twice."""
    by_score = {g: sorted(v, key=lambda c: -c["score"]) for g, v in cases.items()}
    wanted = [("caught", [c for c in by_score["caught"] if not c["facts"]["submission_empty"]][:1]),
              ("caught", [c for c in by_score["caught"] if c["facts"]["submission_empty"]][:1]),
              ("missed", by_score["missed"][-1:]),
              ("missed", [c for c in by_score["missed"] if c["facts"]["tests_after_last_edit"] > 0][:1]),
              ("false alarm", by_score["false alarm"][:3]),
              ("accepted, fixed", by_score["accepted, fixed"][-1:])]
    chosen, seen = [], set()
    for group, found in wanted:
        for case in found:
            if case["id"] not in seen:
                seen.add(case["id"])
                chosen.append((group, case))
    return chosen


def card(group: str, case: dict, entry: str) -> str:
    v = case["verdict"]
    url = html.escape(f"{S3}/bash-only/{entry}/trajs/{case['id']}/{case['id']}.traj.json", quote=True)
    lines = "".join(f"<li class=f>{html.escape(t)}</li>" for t in v["towards_flag"][:3])
    lines += "".join(f"<li class=a>{html.escape(t)}</li>" for t in v["towards_accept"][:2])
    note = case["note"][:240] + ("…" if len(case["note"]) > 240 else "")
    return (f"<article class='{group.split(',')[0].replace(' ', '-')}'><header><b>{html.escape(group)}</b>"
            f"<code>{html.escape(case['id'])}</code></header>"
            f"<p class=v>{'FLAG' if v['flag'] else 'ACCEPT'} · score {v['score']:.2f} (threshold {v['threshold']:.2f})"
            f" · hidden tests: {'not fixed' if case['false_done'] else 'fixed'}</p><ul>{lines}</ul>"
            f"<blockquote>{html.escape(note)}</blockquote><a href='{url}'>full run file</a></article>")


def judge_sentence(path: Path = Path("eval/results/judges_final.json")) -> str:
    """One sentence on the AI judges on GLM-4.5, each read at its threshold fixed on building data."""
    if not path.exists():
        return "The AI-judge baselines have frozen prompts but have not been run yet."
    j = json.loads(path.read_text())
    parts = [f"reading {what}, it caught {pct(j[k]['judge']['fixed_threshold']['catch'])} with "
             f"{pct(j[k]['judge']['fixed_threshold']['false_alarm'])} false alarms (AUROC {j[k]['judge']['auroc']:.3f}, "
             f"about {j[k]['seconds_per_run']:.0f} s per run)"
             for k, what in (("notes", "only the agent's notes"), ("task_diff", "the bug report and the change"))]
    return (f"A free local AI judge ({html.escape(j['judge_freeze']['judge_model'])}, run on a laptop), at a threshold fixed "
            f"in advance on building data the same way: " + "; ".join(parts) + ".")


def main() -> None:
    artifact = json.loads(Path("models/donecheck-v1.json").read_text())
    final = json.loads(Path("eval/results/final_glm45.json").read_text())
    building = json.loads(Path("eval/results/building.json").read_text())
    predictions = read_predictions()
    cases: dict[str, list[dict]] = {g: [] for g in GROUPS.values()}
    for run in iter_runs("data_final"):
        p = predictions[run.instance_id]
        facts = check_run(run)
        cases[GROUPS[(int(p["flagged"]), int(p["false_done"]))]].append(
            {"id": run.instance_id, "score": float(p["score"]), "false_done": p["false_done"] == "1", "facts": facts,
             "verdict": explain(artifact, facts), "note": run.steps[-1].note})
    d, entry = final["donecheck"], ENTRIES["final"][0]
    ft = d["fixed_threshold"]
    folds = "".join(f"<tr><td>{f['held_out']}</td><td>{f['auroc']:.3f}</td><td>{pct(f['with_threshold']['catch'])}</td>"
                    f"<td>{pct(f['with_threshold']['false_alarm'])}</td></tr>"
                    for f in building["donecheck_leave_one_model_out"]["folds"])
    counts = " · ".join(f"{g}: {len(v)}" for g, v in cases.items())
    cards = "".join(card(g, c, entry) for g, c in pick(cases))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.format(
        catch=pct(ft["catch"]), catch_lo=pct(d["fixed_threshold_catch_95"][0]), catch_hi=pct(d["fixed_threshold_catch_95"][1]),
        fa=pct(ft["false_alarm"]), fa_hi=pct(d["fixed_threshold_false_alarm_95"][1]), runs=d["runs"], false_done=d["false_done"],
        auroc=f"{d['auroc']:.3f}", auroc_lo=f"{d['auroc_95'][0]:.3f}", auroc_hi=f"{d['auroc_95'][1]:.3f}",
        roc=pct(d["roc_catch_at_5"]), ms=f"{1000 * final['cost']['donecheck_seconds_per_run']:.1f}",
        agent_cost=f"{final['cost']['agent_usd_per_run_mean']:.3f}", folds=folds, counts=counts, cards=cards,
        commit=final["freeze"]["commit"][:7], judges=judge_sentence()), encoding="utf-8")
    print(f"wrote {OUT}")


TEMPLATE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>DoneCheck demo</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#66645e;--card:#fff;--line:#e4e1d8;--flag:#b3261e;--ok:#1e6b3a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--muted:#a3a19a;--card:#1f1f1d;--line:#34332f;--flag:#ff8a80;--ok:#7fd49b}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:32px 16px}} h1{{font-size:28px;margin:0 0 4px}}
.lead{{color:var(--muted);margin:0 0 24px}} .big{{font-size:20px;border-left:4px solid var(--flag);padding:8px 14px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px;font-size:14px}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
article{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 14px}}
article header{{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}} code{{font-size:13px;color:var(--muted)}}
.v{{margin:6px 0;font-weight:600}} li.f::marker{{content:"▲ ";color:var(--flag)}} li.a::marker{{content:"▼ ";color:var(--ok)}}
blockquote{{margin:8px 0;padding-left:12px;border-left:3px solid var(--line);color:var(--muted);font-size:14px;overflow-wrap:anywhere}}
a{{color:inherit}} .small{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<h1>DoneCheck</h1><p class=lead>Is a coding agent's "done" backed by evidence in its run record? Scored against SWE-bench Verified's hidden tests.</p>
<p class=big>On {runs} runs from GLM-4.5, a model it never saw (on the same 500 bugs), DoneCheck flagged <b>{catch}</b> of the {false_done} false "done" claims
(95% interval {catch_lo} to {catch_hi}) while flagging <b>{fa}</b> of fixed runs (up to {fa_hi}), for $0 in API fees and about {ms} ms per run.
Trusting the agent catches 0%.</p>
<p>The threshold was frozen on building data to aim for at most 5% false alarms on a new model; on GLM-4.5 it turned out stricter.
AUROC {auroc} ({auroc_lo} to {auroc_hi}). Had the threshold been picked on GLM-4.5 itself for at most 5% false alarms, it would have caught {roc}:
that is a point on the curve, not a result anyone could have had in advance. {judges}
The agent itself spent ${agent_cost} per run on average.</p>
<h2>Building data, one model left out at a time</h2>
<table><tr><th>Left-out model</th><th>AUROC</th><th>Caught</th><th>False alarms</th></tr>{folds}</table>
<h2>Real GLM-4.5 runs</h2><p class=small>{counts}. Picked by fixed rules (see src/donecheck/demo.py).
▲ pushed towards flag, ▼ towards accept. Quotes are the agent's last note, cut at 240 characters.</p>
{cards}
<p class=small>Frozen at commit {commit}. Run files are linked, not copied. Method and limits: docs/report.md.</p>
</main></body></html>
"""

if __name__ == "__main__":
    main()
