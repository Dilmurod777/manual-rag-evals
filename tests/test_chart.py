"""The committed chart must match the committed results."""

import json
from pathlib import Path

from eval.charts import rank_counts

ROOT = Path(__file__).resolve().parent.parent


def test_committed_rank_chart_matches_the_results():
    data = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    sidecar = json.loads((ROOT / "docs" / "rank-data.json").read_text(encoding="utf-8"))
    counts = rank_counts(data)

    assert sidecar == counts, "docs/rank.svg is stale: run `python -m eval.charts`"
    for mode, per_rank in counts.items():
        assert sum(per_rank.values()) == data["questions"], f"{mode} lost a question"
