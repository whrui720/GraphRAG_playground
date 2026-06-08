"""Build an interactive pyvis network from GraphRAG entity/relationship parquets."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd


_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _color_for(value: str, mapping: dict[str, str]) -> str:
    if value not in mapping:
        mapping[value] = _PALETTE[len(mapping) % len(_PALETTE)]
    return mapping[value]


def build_pyvis_html(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    *,
    max_nodes: int = 40,
    min_degree: int = 1,
    highlight: str | None = None,
) -> str:
    """Return self-contained HTML for an interactive knowledge graph.

    Node size scales with `degree`; color encodes `type`. Nodes whose title
    matches `highlight` (case-insensitive substring) are bordered in red.
    """
    from pyvis.network import Network

    ents = entities.copy()
    if "degree" in ents.columns:
        ents = ents[ents["degree"] >= min_degree]
        ents = ents.sort_values("degree", ascending=False)
    ents = ents.head(max_nodes)

    keep = set(ents["title"].astype(str))
    rels = relationships[
        relationships["source"].astype(str).isin(keep)
        & relationships["target"].astype(str).isin(keep)
    ]

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.force_atlas_2based(gravity=-50, spring_length=120)

    color_map: dict[str, str] = {}
    highlight_lc = (highlight or "").strip().lower()

    for _, row in ents.iterrows():
        title = str(row["title"])
        etype = str(row.get("type", "ENTITY"))
        degree = int(row.get("degree", 1) or 1)
        desc = str(row.get("description", ""))[:300]
        size = 10 + min(degree, 50) * 1.5
        color = _color_for(etype, color_map)
        is_match = bool(highlight_lc) and highlight_lc in title.lower()
        net.add_node(
            title,
            label=title,
            title=f"{etype} (deg {degree})\n{desc}",
            color={"background": color, "border": "#d62728" if is_match else "#333"},
            borderWidth=4 if is_match else 1,
            size=size,
        )

    for _, row in rels.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        weight = float(row.get("weight", 1.0) or 1.0)
        desc = str(row.get("description", ""))[:200]
        net.add_edge(src, tgt, value=weight, title=desc)

    # pyvis 0.3+ requires writing to disk; capture the HTML and return it.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.write_html(str(tmp_path), notebook=False, open_browser=False)
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
