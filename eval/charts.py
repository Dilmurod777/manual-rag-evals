"""Draw the result: which rank the gold page got, hit rate by chapter, wrong citations."""

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results.json"
DOCS = ROOT / "docs"

INK, ACCENT, MUTED, RULE = "#15171a", "#b8321a", "#5d5f5a", "#b6b5ad"
MODES = ("vector", "hybrid")


def bars(categories, series, denom, title, subtitle, footer, height=250):
    width, left, top, plot_h = 900, 52, 84, 120
    slot = (width - left - 30) / len(categories)
    bar_w = min(slot * 0.30, 46)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Archivo,Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="15" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="{left}" y="43" font-size="12" fill="{MUTED}">{subtitle}</text>',
        f'<rect x="{left}" y="57" width="10" height="10" fill="{INK}"/>',
        f'<text x="{left + 15}" y="66" font-size="11" fill="{MUTED}">A · vector only</text>',
        f'<rect x="{left + 110}" y="57" width="10" height="10" fill="{ACCENT}"/>',
        f'<text x="{left + 125}" y="66" font-size="11" fill="{MUTED}">B · hybrid</text>',
    ]
    for i, cat in enumerate(categories):
        x0 = left + i * slot + (slot - bar_w * 2 - 4) / 2
        for k, mode in enumerate(MODES):
            value = series[mode].get(cat, 0)
            h = plot_h * value / denom[cat]
            out.append(
                f'<rect x="{x0 + k * (bar_w + 4):.1f}" y="{top + plot_h - h:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{INK if mode == "vector" else ACCENT}"/>'
            )
            out.append(
                f'<text x="{x0 + k * (bar_w + 4) + bar_w / 2:.1f}" y="{top + plot_h - h - 5:.1f}" '
                f'font-size="10" text-anchor="middle" fill="{MUTED}">{value}</text>'
            )
        out.append(
            f'<text x="{x0 + bar_w + 2:.1f}" y="{top + plot_h + 17}" font-size="11" '
            f'text-anchor="middle" fill="{MUTED}">{cat}</text>'
        )
    out += [
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 24}" y2="{top + plot_h}" stroke="{RULE}"/>',
        f'<text x="{left}" y="{height - 18}" font-size="10" fill="{MUTED}">{footer}</text>',
        "</svg>",
    ]
    return "\n".join(out)


def rank_chart(data):
    series = {mode: {} for mode in MODES}
    for mode in MODES:
        for row in data[mode]["rows"]:
            key = str(row["rank"]) if row["rank"] else "miss"
            series[mode][key] = series[mode].get(key, 0) + 1
    categories = ["1", "2", "3", "4", "5", "miss"]
    total = len(data["vector"]["rows"])
    return bars(
        categories,
        series,
        {cat: total for cat in categories},
        "Which rank the gold page got",
        f"{total} questions, top five paragraphs, one per page",
        "rank of the gold page in the top five · miss = not retrieved at all",
    )


def chapter_chart(data):
    rows = data["vector"]["rows"]
    chapters = [str(c) for c in sorted({int(r["gold"].split("-")[0]) for r in rows})]
    series = {mode: {} for mode in MODES}
    denom = {}
    for chapter in chapters:
        denom[chapter] = sum(1 for r in rows if r["gold"].split("-")[0] == chapter)
        for mode in MODES:
            series[mode][chapter] = sum(
                1
                for r in data[mode]["rows"]
                if r["gold"].split("-")[0] == chapter and r["rank"]
            )
    return bars(
        chapters,
        series,
        denom,
        "hit@5 by chapter",
        "share of that chapter's questions whose gold page made the top five",
        "chapter number in AC 43.13-1B · counts are hits out of that chapter's questions",
    )


def wrong_citations(data):
    lines = ["| question | gold page | page the answer cited |", "|---|---|---|"]
    for row in data["hybrid"]["rows"]:
        if not row["right"]:
            lines.append(f"| {row['q']} | {row['gold']} | {row['cited'] or 'none'} |")
    return "\n".join(lines)


def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "rank.svg").write_text(rank_chart(data), encoding="utf-8")
    (DOCS / "by_chapter.svg").write_text(chapter_chart(data), encoding="utf-8")
    for mode in MODES:
        rows = data[mode]["rows"]
        print(
            f"{mode:<7} median retrieval {median(r['retrieval_ms'] for r in rows):.0f} ms  "
            f"median answer {median(r['answer_ms'] for r in rows):.0f} ms  "
            f"cited {sum(r['right'] for r in rows)}/{len(rows)}"
        )
    print("\n" + wrong_citations(data))


if __name__ == "__main__":
    main()
