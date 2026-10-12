"""Figures render offline, in both themes, deterministically (no git churn on rebuild)."""

import json

import numpy as np
import pandas as pd
import pytest

from rag.viz import figures as F


def assert_pair(paths):
    light, dark = paths
    assert light.name.endswith("-light.svg") and dark.name.endswith("-dark.svg")
    a, b = light.read_text(encoding="utf-8"), dark.read_text(encoding="utf-8")
    assert a.lstrip().startswith("<?xml") and "<svg" in a
    assert F.LIGHT.surface in a and F.DARK.surface in b and a != b


@pytest.mark.parametrize("fn", [F.bm25_saturation, F.rrf_rank_weight, F.maxsim_heatmap, F.ivf_partition])
def test_formula_figures(tmp_path, fn):
    assert_pair(fn(tmp_path))


def test_rendering_is_deterministic(tmp_path):
    first = [p.read_bytes() for p in F.rrf_rank_weight(tmp_path / "a")]
    second = [p.read_bytes() for p in F.rrf_rank_weight(tmp_path / "b")]
    assert first == second


def test_chunk_windows(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text(("Paragraph text. " * 40 + "\n\n") * 4, encoding="utf-8")
    assert_pair(F.chunk_windows(doc, out_dir=tmp_path))


def test_chunk_embeddings_pca(tmp_path):
    idx = tmp_path / "idx"
    idx.mkdir()
    np.save(idx / "embeddings.npy", np.random.default_rng(0).normal(size=(9, 16)))
    meta = [{"text": "t", "source": src, "chunk_id": i % 3}
            for i, src in enumerate(["ancient_rome.txt"] * 3 + ["ai_history.txt"] * 3 + ["climate_science.txt"] * 3)]
    (idx / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")
    assert_pair(F.chunk_embeddings_pca(idx, out_dir=tmp_path))


def test_graphrag_components(tmp_path):
    frames = {
        "documents": pd.DataFrame({"id": ["d1", "d2"], "title": ["ancient_rome.txt", "ai_history.txt"]}),
        "text_units": pd.DataFrame({"id": ["u1", "u2"], "document_id": ["d1", "d2"]}),
        "entities": pd.DataFrame({"id": ["e1", "e2", "e3"], "title": ["CAESAR", "POMPEY", "HINTON"],
                                  "text_unit_ids": [["u1"], ["u1"], ["u2"]]}),
        "relationships": pd.DataFrame({"source": ["CAESAR"], "target": ["POMPEY"]}),
        "communities": pd.DataFrame({"entity_ids": [["e1", "e2"]]}),
    }
    assert_pair(F.graphrag_components(out_dir=tmp_path, frames=frames))


def test_vectorless_tree(tmp_path):
    index = {"index": [{"node_id": "d0_001", "title": "Rise of Caesar", "summary": "", "start_line": 1,
                        "end_line": 4, "source": "ancient_rome.txt",
                        "children": [{"node_id": "d0_001.1", "title": "Birth", "summary": "", "start_line": 1,
                                      "end_line": 2, "source": "ancient_rome.txt", "children": []}]}],
             "doc_lines": {"ancient_rome.txt": ["a", "b", "c", "d"]}}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    assert_pair(F.vectorless_tree(path, out_dir=tmp_path))


def test_benchmark_overview(tmp_path):
    by_type = {"factual": {"answer_correctness": 0.9}, "thematic": {"answer_correctness": 0.0}}
    output = {"meta": {"run_id": "r1"},
              "summary": {"standard_rag": {"by_type": by_type, "latency_s_p50": 2.0},
                          "graphrag": {"by_type": {"factual": {"answer_correctness": None}}, "latency_s_p50": 9.0}}}
    assert_pair(F.benchmark_overview(output, tmp_path))
