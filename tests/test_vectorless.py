import json

import pytest

from rag.utils import parse_llm_json
from rag.vectorless_rag import DocumentNode, VectorlessRAG

TREE = [
    {
        "node_id": "001", "title": "Rise", "summary": "Early career.",
        "start_line": 1, "end_line": 4,
        "children": [
            {"node_id": "001.1", "title": "Birth", "summary": "Born 100 BC.",
             "start_line": 1, "end_line": 2, "children": []},
            {"node_id": "001.2", "title": "Triumvirate", "summary": "Alliance.",
             "start_line": 3, "end_line": 4, "children": []},
        ],
    }
]
LINES = ["line one", "line two", "line three", "line four"]


@pytest.fixture
def rag():
    """VectorlessRAG with an in-memory index and no API client."""
    r = VectorlessRAG.__new__(VectorlessRAG)
    r._model = "fake"
    r._usage = [0, 0]
    r._doc_lines = {"rome.txt": LINES}
    r._index = r._parse_nodes(TREE, "rome.txt", prefix="d0")
    return r


def test_parse_llm_json_strips_code_fences():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('  {"a": 2} ') == {"a": 2}


def test_node_ids_are_prefixed_per_document(rag):
    ids = list(rag._node_map(rag._index))
    assert ids == ["d0_001", "d0_001.1", "d0_001.2"]


def test_node_text_uses_inclusive_one_indexed_line_ranges(rag):
    node = rag._node_map(rag._index)["d0_001.2"]
    assert rag._node_text(node) == "line three\nline four"


def test_dict_round_trip_preserves_tree():
    node = DocumentNode.from_dict({**TREE[0], "source": "rome.txt",
                                   "children": [{**c, "source": "rome.txt"} for c in TREE[0]["children"]]})
    again = DocumentNode.from_dict(node.to_dict())
    assert again.to_dict() == node.to_dict()
    assert "start_line" not in json.dumps(node.to_retrieval_view())


def test_retrieve_maps_llm_node_list_to_text_and_skips_unknown_ids(rag):
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return json.dumps({"thinking": "need birth", "node_list": ["d0_001.1", "bogus"]})

    rag._generate = fake_generate
    nodes, thinking = rag.retrieve("When was he born?")

    assert thinking == "need birth"
    assert [(n.node_id, n.text) for n in nodes] == [("d0_001.1", "line one\nline two")]
    # retrieval prompt shows titles/summaries, never raw text
    assert "Born 100 BC." in prompts[0] and "line one" not in prompts[0]


def test_retrieve_caps_at_max_nodes(rag):
    rag._generate = lambda p: json.dumps({"thinking": "", "node_list": ["d0_001", "d0_001.1", "d0_001.2"]})
    nodes, _ = rag.retrieve("q", max_nodes=2)
    assert [n.node_id for n in nodes] == ["d0_001", "d0_001.1"]
