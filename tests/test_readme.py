"""The README quotes counts; this fails when the two drift apart."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_quotes_every_measured_count():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("results.json", "results-blind.json", "results-paraphrase.json"):
        path = ROOT / "eval" / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        n = data["questions"]
        for mode in ("vector", "hybrid", "rerank"):
            hits = round(data[mode]["hit_at_5"] * n)
            assert f"{hits}/{n}" in readme, (
                f"{name}: {mode} {hits}/{n} is not in the README"
            )
