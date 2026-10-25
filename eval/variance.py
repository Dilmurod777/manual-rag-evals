"""Three runs of the same 60 questions: how much of a number is the run, not the system."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("vector", "hybrid", "rerank")


def main():
    runs = [
        json.loads((HERE / f"variance-{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)
    ]
    n = runs[0]["questions"]
    out = {"questions": n, "runs": 3, "modes": {}}
    for mode in MODES:
        hits = [round(run[mode]["hit_at_5"] * n) for run in runs]
        mrr = [run[mode]["mrr"] for run in runs]
        cited = [round(run[mode]["citation"] * n) for run in runs]
        out["modes"][mode] = {
            "hit_at_5": hits,
            "mrr": mrr,
            "citation": cited,
        }
        print(f"{mode:<7} hit@5 {hits}  mrr {min(mrr):.3f}-{max(mrr):.3f}  cited {cited}")
    (HERE / "variance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
