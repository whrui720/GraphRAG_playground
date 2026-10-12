"""Figures for the Obsidian notes, generated from formulas and real repo data.

Every figure is written twice, `<name>-light.svg` and `<name>-dark.svg`, each
with its own palette steps. The vault's CSS snippet (.obsidian/snippets/
rag-vault.css) shows the one matching the Obsidian theme. Notes embed both:

    ![[bm25_saturation-light.svg]]![[bm25_saturation-dark.svg]]

Regenerate everything with `python -m rag.viz.figures`. Figures that need local
artifacts (embeddings, GraphRAG parquet, vectorless index, benchmark JSON) are
skipped if those are missing; the committed SVGs stay as they were.

Palette and mark specs follow the dataviz reference palette: categorical slots
1-3 (validated all-pairs in both modes), hairline grids, 2px lines, legends for
2+ series, text in ink colors, never series colors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "notes" / "assets" / "figures"


# ── Theme ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]   # categorical slots 1-3: blue, orange, aqua
    seq: tuple[str, str]           # sequential ramp: near-zero -> high


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
              ("#2a78d6", "#eb6834", "#1baf7a"), ("#cde2fb", "#104281"))
DARK = Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
             ("#3987e5", "#d95926", "#199e70"), ("#184f95", "#cde2fb"))
THEMES = (LIGHT, DARK)

# Color follows the entity: each document / variant keeps its slot in every figure.
DOC_SLOT = {"ancient_rome.txt": 0, "ai_history.txt": 1, "climate_science.txt": 2}
DOC_LABEL = {"ancient_rome.txt": "Rome", "ai_history.txt": "AI history", "climate_science.txt": "Climate"}
VARIANT_SLOT = {"standard_rag": 0, "vectorless": 1, "graphrag": 2}
VARIANT_LABEL = {"standard_rag": "Naive RAG", "vectorless": "Vectorless RAG", "graphrag": "GraphRAG"}

PX = 1 / 96 * 72  # one CSS pixel in points (SVG units)


def _rc(t: Theme) -> dict:
    return {
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "text.color": t.ink, "axes.labelcolor": t.ink2, "axes.titlecolor": t.ink,
        "xtick.color": t.muted, "ytick.color": t.muted, "xtick.labelcolor": t.ink2, "ytick.labelcolor": t.ink2,
        "axes.edgecolor": t.axis, "axes.linewidth": PX, "grid.color": t.grid, "grid.linewidth": PX,
        "xtick.major.width": PX, "ytick.major.width": PX, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": t.ink2,
        "lines.linewidth": 2 * PX, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "svg.fonttype": "path", "svg.hashsalt": "rag-vault",
    }


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(axis == "y" and side == "bottom" or axis == "x" and side == "left")


def _ink_on(color: str, t: Theme) -> str:
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.5 else "#fcfcfb"


def _end_label(ax, x, y, text, t: Theme, dx: float = 4) -> None:
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=t.ink2)


def render(name: str, draw: Callable[[Theme], plt.Figure], out_dir: Path = FIG_DIR) -> list[Path]:
    """Draw a figure in both themes and save `<name>-light.svg` / `<name>-dark.svg`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in THEMES:
        with plt.rc_context(_rc(t)):
            fig = draw(t)
            path = out_dir / f"{name}-{t.mode}.svg"
            fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
            plt.close(fig)
        paths.append(path)
    return paths


# ── Formula figures ───────────────────────────────────────────────────────────

def bm25_saturation(out_dir: Path = FIG_DIR) -> list[Path]:
    """BM25's term-frequency saturation (k1) and length normalization (b)."""

    def tf_part(f, k1, b=0.75, ratio=1.0):
        return f * (k1 + 1) / (f + k1 * (1 - b + b * ratio))

    def draw(t: Theme):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.42})
        f = np.linspace(0, 10, 200)
        for i, k1 in enumerate([0.5, 1.2, 2.0]):
            y = tf_part(f, k1)
            a1.plot(f, y, color=t.series[i], label=f"k₁ = {k1}")
            _end_label(a1, f[-1], y[-1], f"k₁ = {k1}", t)
        a1.set(title="Repeats have diminishing returns", xlabel="term frequency f(q, D)",
               ylabel="tf component", xlim=(0, 10), ylim=(0, 3.2))
        _grid(a1)

        ratio = np.linspace(0.25, 3, 200)
        for i, b in enumerate([0.0, 0.75, 1.0]):
            y = tf_part(3, 1.2, b, ratio)
            a2.plot(ratio, y, color=t.series[i], label=f"b = {b}")
        a2.set(title="Long documents are discounted", xlabel="|D| / avgdl   (f = 3, k₁ = 1.2)",
               ylabel="tf component", xlim=(0.25, 3), ylim=(0, 2.2))
        _grid(a2)
        a1.legend(loc="lower right")
        a2.legend(loc="lower left")
        return fig

    return render("bm25_saturation", draw, out_dir)


def rrf_rank_weight(out_dir: Path = FIG_DIR) -> list[Path]:
    """How the RRF constant k flattens the advantage of the top ranks."""

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(6.4, 2.7))
        r = np.arange(1, 21)
        for i, k in enumerate([0, 10, 60]):
            y = (1 / (k + r)) / (1 / (k + 1))
            ax.plot(r, y, color=t.series[i], marker="o", markersize=4, markeredgecolor=t.surface,
                    markeredgewidth=1.5 * PX, label=f"k = {k}")
            _end_label(ax, r[-1], y[-1], f"k = {k}", t, dx=6)
        ax.set(title="RRF weight relative to rank 1:  (1 + k) / (rank + k)", xlabel="rank in one list",
               ylabel="relative weight", xlim=(0.5, 20.5), ylim=(0, 1.05), xticks=[1, 5, 10, 15, 20])
        _grid(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        return fig

    return render("rrf_rank_weight", draw, out_dir)


def maxsim_heatmap(out_dir: Path = FIG_DIR) -> list[Path]:
    """The ColBERT worked example: per-query-token max over document tokens."""
    q_tokens = ["capital", "France"]
    d_tokens = ["Paris", "is", "the", "capital", "of", "France"]
    sim = np.array([[0.61, 0.10, 0.08, 0.93, 0.12, 0.40],
                    [0.78, 0.05, 0.07, 0.35, 0.15, 0.96]])

    def draw(t: Theme):
        cmap = LinearSegmentedColormap.from_list("seq", list(t.seq))
        fig, ax = plt.subplots(figsize=(6.4, 1.9))
        for (i, j), v in np.ndenumerate(sim):
            color = cmap(v)
            ax.add_patch(Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, color=color, lw=0))
            is_max = j == sim[i].argmax()
            ax.text(j + 0.5, i + 0.5, f"{v:.2f}" + ("  max" if is_max else ""), ha="center", va="center",
                    fontsize=9, fontweight="bold" if is_max else "normal", color=_ink_on(matplotlib.colors.to_hex(color), t))
        ax.set_xlim(0, len(d_tokens))
        ax.set_ylim(len(q_tokens), 0)
        ax.set_xticks(np.arange(len(d_tokens)) + 0.5, d_tokens)
        ax.set_yticks(np.arange(len(q_tokens)) + 0.5, q_tokens)
        ax.xaxis.tick_top()
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("MaxSim (illustrative values):  S = 0.93 + 0.96 = 1.89", pad=22)
        ax.set_xlabel("document tokens  →  each query token keeps its best match", labelpad=6)
        return fig

    return render("maxsim_heatmap", draw, out_dir)


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    centroids = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((x[:, None] - centroids[None]) ** 2).sum(-1), axis=1)
        centroids = np.array([x[labels == c].mean(0) if np.any(labels == c) else centroids[c] for c in range(k)])
    return centroids


def ivf_partition(out_dir: Path = FIG_DIR) -> list[Path]:
    """IVF on synthetic 2-D data: k-means cells, the query searches only nprobe of them."""
    rng = np.random.default_rng(7)
    centers = rng.uniform(-4, 4, size=(6, 2))
    x = np.vstack([c + rng.normal(scale=0.9, size=(60, 2)) for c in centers])
    cents = _kmeans(x, 8, rng)
    q = np.array([0.6, 0.4])
    probed = np.argsort(((cents - q) ** 2).sum(1))[:2]
    labels = np.argmin(((x[:, None] - cents[None]) ** 2).sum(-1), axis=1)

    gx, gy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
    grid_lab = np.argmin(((np.stack([gx, gy], -1)[..., None, :] - cents) ** 2).sum(-1), axis=-1)

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.contour(gx, gy, grid_lab, levels=np.arange(9) - 0.5, colors=t.axis, linewidths=PX)
        inside = np.isin(labels, probed)
        ax.scatter(*x[~inside].T, s=10, color=t.muted, alpha=0.55, lw=0, label="vectors skipped")
        ax.scatter(*x[inside].T, s=16, color=t.series[0], lw=0, label="vectors scanned (nprobe = 2)")
        ax.scatter(*cents.T, s=40, marker="x", color=t.ink2, linewidths=2 * PX, label="centroids (nlist = 8)")
        ax.scatter(*q, s=90, color=t.series[1], edgecolor=t.surface, linewidths=2 * PX, zorder=5, label="query")
        ax.set(xlim=(-7, 7), ylim=(-7, 7), xticks=[], yticks=[], aspect="equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("IVF: only the cells nearest the query are scanned")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return fig

    return render("ivf_partition", draw, out_dir)


# ── Repo-data figures ─────────────────────────────────────────────────────────

def chunk_windows(text_path: Path = ROOT / "data" / "sample_texts" / "ancient_rome.txt",
                  chunk_size: int = 600, overlap: int = 100, out_dir: Path = FIG_DIR) -> list[Path]:
    """TextChunker's windows over a real document, against its paragraph breaks."""
    from rag.standard_rag import TextChunker

    text = Path(text_path).read_text(encoding="utf-8")
    step = chunk_size - overlap
    chunks = TextChunker(chunk_size, overlap).chunk_text(text)
    spans = [(i * step, min(i * step + chunk_size, len(text))) for i in range(len(chunks))]
    breaks = [i for i in range(len(text)) if text.startswith("\n\n", i)]

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(spans) + 1.0))
        for b in breaks:
            ax.axvline(b, color=t.axis, lw=PX, zorder=0)
        for row, (s, e) in enumerate(spans):
            ax.add_patch(FancyBboxPatch((s, row - 0.28), e - s, 0.56, boxstyle="round,pad=0,rounding_size=0.12",
                                        mutation_aspect=1 / 60, color=t.series[0], lw=0))
            ax.text(e + 25, row, f"#{row}  {e - s} chars", va="center", fontsize=8, color=t.ink2)
        ax.set(xlim=(0, len(text) + 420), ylim=(len(spans) - 0.4, -0.8), yticks=[],
               xlabel=f"character offset in {Path(text_path).name}")
        ax.set_title(f"TextChunker({chunk_size}, {overlap}): fixed windows ignore paragraph breaks (gray lines)")
        _grid(ax, "x")
        ax.grid(False)
        return fig

    return render("chunk_windows", draw, out_dir)


def chunk_embeddings_pca(index_dir: Path = ROOT / "output" / "rag_index", out_dir: Path = FIG_DIR) -> list[Path]:
    """The real chunk embeddings projected to 2-D with PCA, colored by source document."""
    emb = np.load(Path(index_dir) / "embeddings.npy")
    meta = json.loads((Path(index_dir) / "chunks.json").read_text(encoding="utf-8"))
    centered = emb - emb.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    var = (s ** 2 / (s ** 2).sum())[:2]
    sources = [m["source"] for m in meta]

    def draw(t: Theme):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for doc, slot in DOC_SLOT.items():
            idx = [i for i, src in enumerate(sources) if src == doc]
            if not idx:
                continue
            ax.scatter(*xy[idx].T, s=46, color=t.series[slot], edgecolor=t.surface,
                       linewidths=2 * PX, label=f"{DOC_LABEL[doc]} ({len(idx)} chunks)", zorder=3)
        ax.set(xlabel=f"PC1 ({var[0]:.0%} of variance)", ylabel=f"PC2 ({var[1]:.0%})", xticks=[], yticks=[])
        ax.set_title(f"{len(sources)} chunk embeddings ({emb.shape[1]}-d) projected to 2-D")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _grid(ax, "both")
        ax.spines["left"].set_visible(True)
        return fig

    return render("chunk_embeddings_pca", draw, out_dir)


def graphrag_components(output_dir: Path = ROOT / "ragtest" / "output", out_dir: Path = FIG_DIR,
                        frames: dict | None = None) -> list[Path]:
    """The extracted entity graph: which entities ended up in a community, per document.

    `frames` (entities, relationships, text_units, documents, communities DataFrames)
    can be passed directly, mainly for tests.
    """
    import networkx as nx
    import pandas as pd

    if frames is None:
        frames = {n: pd.read_parquet(Path(output_dir) / f"{n}.parquet")
                  for n in ["entities", "relationships", "text_units", "documents", "communities"]}
    ent, rel, tu, docs, com = (frames[n] for n in ["entities", "relationships", "text_units", "documents", "communities"])

    doc_title = dict(zip(docs["id"], docs["title"]))
    unit_doc = {u: doc_title.get(d) for u, d in zip(tu["id"], tu["document_id"])}
    clustered = set()
    for ids in com["entity_ids"]:
        clustered.update(ids)

    G = nx.Graph()
    for _, e in ent.iterrows():
        units = list(e["text_unit_ids"])
        doc = max(set(unit_doc.get(u) for u in units), key=[unit_doc.get(u) for u in units].count) if units else None
        G.add_node(e["title"], doc=doc, clustered=e["id"] in clustered)
    G.add_edges_from(zip(rel["source"], rel["target"]))
    pos = nx.spring_layout(G, seed=3, k=0.35, iterations=200)

    counts = {d: [0, 0] for d in DOC_SLOT}
    for _, data in G.nodes(data=True):
        if data["doc"] in counts:
            counts[data["doc"]][0 if data["clustered"] else 1] += 1

    def draw(t: Theme):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.25})
        nx.draw_networkx_edges(G, pos, ax=a1, edge_color=t.axis, width=PX)
        for doc, slot in DOC_SLOT.items():
            for flag in (True, False):
                nodes = [n for n, d in G.nodes(data=True) if d["doc"] == doc and d["clustered"] == flag]
                if nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=a1, node_size=22 if flag else 16,
                                           node_color=t.series[slot] if flag else t.surface,
                                           edgecolors=t.surface if flag else t.series[slot],
                                           linewidths=1.5 * PX if flag else 1.5)
        a1.set_axis_off()
        a1.set_title("Entity graph: filled = in a community")

        y = np.arange(len(DOC_SLOT))
        for i, (doc, slot) in enumerate(DOC_SLOT.items()):
            inside, outside = counts[doc]
            a2.barh(i, inside, height=0.5, color=t.series[slot], lw=0)
            a2.barh(i, outside, left=inside + 0.6, height=0.5, color=t.series[slot], alpha=0.25, lw=0)
            a2.text(inside + outside + 2, i, f"{inside} / {inside + outside}", va="center", fontsize=8.5, color=t.ink2)
        a2.set_yticks(y, [DOC_LABEL[d] for d in DOC_SLOT])
        a2.invert_yaxis()
        a2.set(xlabel="entities (solid = in a community)", xlim=(0, max(sum(c) for c in counts.values()) * 1.35))
        a2.set_title("Clustered entities per document")
        _grid(a2, "x")
        return fig

    return render("graphrag_components", draw, out_dir)


def vectorless_tree(index_path: Path = ROOT / "output" / "vectorless_index.json", out_dir: Path = FIG_DIR) -> list[Path]:
    """The LLM-authored ToC tree as an icicle: each node spans its line range."""
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    docs = sorted({n["source"] for n in data["index"]}, key=lambda d: DOC_SLOT.get(d, 9))

    def draw(t: Theme):
        fig, axes = plt.subplots(len(docs), 1, figsize=(7.2, 1.15 * len(docs) + 0.4), gridspec_kw={"hspace": 0.9})
        axes = np.atleast_1d(axes)
        for ax, doc in zip(axes, docs):
            n_lines = len(data["doc_lines"].get(doc, [])) or 1
            color = t.series[DOC_SLOT.get(doc, 0)]

            def place(nodes, depth):
                for n in nodes:
                    s, e = n["start_line"] - 1, n["end_line"]
                    fill = color if depth == 0 else matplotlib.colors.to_hex(
                        np.array(to_rgb(color)) * 0.35 + np.array(to_rgb(t.surface)) * 0.65)
                    ax.add_patch(Rectangle((s + 0.06, depth + 0.06), e - s - 0.12, 0.88, color=fill, lw=0))
                    label = n["title"] if depth == 0 else n["node_id"].split("_")[1]
                    max_chars = int((e - s) / n_lines * 95)
                    if len(label) > max_chars:
                        short = label.rsplit(".", 1)[-1]
                        label = (label[: max_chars - 1] + "…" if depth == 0 and max_chars > 3
                                 else short if len(short) <= max(max_chars, 2) else "")
                    ax.text((s + e) / 2, depth + 0.5, label, ha="center", va="center", fontsize=7.5,
                            color=_ink_on(fill, t))
                    place(n["children"], depth + 1)

            place([n for n in data["index"] if n["source"] == doc], 0)
            ax.set(xlim=(0, n_lines), ylim=(2, 0), yticks=[0.5, 1.5], yticklabels=["section", "leaf"])
            ax.set_xticks([0, n_lines], ["line 1", f"line {n_lines}"])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{DOC_LABEL.get(doc, doc)} ({doc})", fontsize=9, pad=4)
        fig.suptitle("Vectorless index: LLM-authored sections and leaves over each document's lines",
                     x=0.125, ha="left", fontsize=10, fontweight="semibold", y=1.0)
        return fig

    return render("vectorless_tree", draw, out_dir)


def benchmark_overview(output: dict, out_dir: Path = FIG_DIR) -> list[Path]:
    """Correctness by question type and median latency, per variant, for one benchmark run."""
    summary = output.get("summary", {})
    variants = [v for v in VARIANT_SLOT if v in summary]
    types = ["factual", "relational", "multi_hop", "thematic"]
    run_id = output.get("meta", {}).get("run_id", "")

    def draw(t: Theme):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.45})
        width = 0.8 / max(len(variants), 1)
        for i, v in enumerate(variants):
            vals = [(summary[v]["by_type"].get(ty) or {}).get("answer_correctness") for ty in types]
            xs = np.arange(len(types)) + (i - (len(variants) - 1) / 2) * width
            a1.bar(xs, [x or 0 for x in vals], width=width * 0.86, color=t.series[VARIANT_SLOT[v]],
                   lw=0, label=VARIANT_LABEL[v])
            for x, val in zip(xs, vals):
                if not val:
                    a1.text(x, 0.015, "n/a" if val is None else "0", ha="center", va="bottom",
                            fontsize=7.5, color=t.muted)
        a1.set_xticks(np.arange(len(types)), [ty.replace("_", "-") for ty in types])
        a1.set(ylim=(0, 1), ylabel="answer correctness (judge)", yticks=[0, 0.25, 0.5, 0.75, 1])
        a1.set_title("Correctness by question type")
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(variants))
        _grid(a1)

        lat = [summary[v].get("latency_s_p50") or 0 for v in variants]
        a2.barh(np.arange(len(variants)), lat, height=0.5, lw=0,
                color=[t.series[VARIANT_SLOT[v]] for v in variants])
        for i, val in enumerate(lat):
            a2.text(val + max(lat) * 0.04, i, f"{val:.1f}s", va="center", fontsize=8.5, color=t.ink2)
        a2.set_yticks(np.arange(len(variants)), [VARIANT_LABEL[v] for v in variants])
        a2.invert_yaxis()
        a2.set(xlim=(0, max(lat + [1]) * 1.3), xlabel="seconds")
        a2.set_title("Median latency")
        _grid(a2, "x")
        fig.text(0.125, -0.12, f"run {run_id} · 10 questions, so differences under ~0.1 are noise",
                 fontsize=7.5, color=t.muted)
        return fig

    return render("benchmark_overview", draw, out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_all(out_dir: Path = FIG_DIR) -> list[Path]:
    """Build every figure whose inputs exist; report the ones skipped."""
    jobs: list[tuple[str, Callable[[], list[Path]], list[Path]]] = [
        ("bm25_saturation", lambda: bm25_saturation(out_dir), []),
        ("rrf_rank_weight", lambda: rrf_rank_weight(out_dir), []),
        ("maxsim_heatmap", lambda: maxsim_heatmap(out_dir), []),
        ("ivf_partition", lambda: ivf_partition(out_dir), []),
        ("chunk_windows", lambda: chunk_windows(out_dir=out_dir), [ROOT / "data" / "sample_texts" / "ancient_rome.txt"]),
        ("chunk_embeddings_pca", lambda: chunk_embeddings_pca(out_dir=out_dir), [ROOT / "output" / "rag_index" / "embeddings.npy"]),
        ("graphrag_components", lambda: graphrag_components(out_dir=out_dir), [ROOT / "ragtest" / "output" / "entities.parquet"]),
        ("vectorless_tree", lambda: vectorless_tree(out_dir=out_dir), [ROOT / "output" / "vectorless_index.json"]),
        ("benchmark_overview", lambda: benchmark_overview(
            json.loads((ROOT / "output" / "benchmark_results.json").read_text(encoding="utf-8")), out_dir),
         [ROOT / "output" / "benchmark_results.json"]),
    ]
    written = []
    for name, job, needs in jobs:
        missing = [p for p in needs if not p.exists()]
        if missing:
            print(f"  skip {name}: missing {missing[0].relative_to(ROOT)}")
            continue
        written += job()
    return written


if __name__ == "__main__":
    for p in build_all():
        print(f"Wrote {p.relative_to(ROOT)}")
