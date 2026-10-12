"""Static checks on the Obsidian vault under notes/ (no Obsidian needed).

- every note has valid YAML frontmatter with a known `type`
- variant/concept notes use the schema in notes/00-index/Templates/variant.md
- every [[wikilink]] (body and frontmatter) resolves to a note, and #headings exist
- every relative Markdown link (e.g. to code) points at an existing file
- Dataview queries only reference fields that the notes actually define
- every benchmarked variant (rag/eval/registry.py) has a note with a Results query
"""

import re
from pathlib import Path

import pytest

from rag.eval.obsidian_export import RESULT_FIELDS, read_frontmatter
from rag.eval.registry import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
TEMPLATES = NOTES / "00-index" / "Templates"

ALL_NOTES = sorted(NOTES.rglob("*.md"))
CONTENT_NOTES = [p for p in ALL_NOTES if TEMPLATES not in p.parents]
VARIANT_NOTES = [p for p in CONTENT_NOTES if (NOTES / "variants") in p.parents]
NOTE_NAMES = {p.stem: p for p in ALL_NOTES}

TYPES = {"rag-variant", "rag-concept", "rag-result", "rag-index"}
FAMILIES = {"baseline", "retrieval-quality", "index", "control-flow", "modality",
            "architecture", "alternative", "cross-cutting"}
LEVELS = {"none", "low", "medium", "high"}
STATUSES = {"idea", "implemented", "benchmarked"}
VARIANT_FIELDS = ["type", "aliases", "family", "adds_latency", "ingest_cost", "query_cost",
                  "requires", "combines_with", "alternatives", "code", "papers", "status", "tags"]
DATAVIEW_IMPLICIT = re.compile(r"^file\.\w+$")

FENCE = re.compile(r"^```.*?^```", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
WIKILINK = re.compile(r"\[\[([^\]]+?)\]\]")
MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
DATAVIEW = re.compile(r"^```dataview\n(.*?)^```", re.S | re.M)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text[text.find("\n---", 3) + 4:]
    return text


def prose(path: Path) -> str:
    """Body without fenced or inline code, where links aren't links."""
    return INLINE_CODE.sub("", FENCE.sub("", body(path)))


def headings(path: Path) -> set[str]:
    return {m.group(1).strip() for m in re.finditer(r"^#+\s+(.+)$", FENCE.sub("", body(path)), re.M)}


def wikilink_targets(text: str) -> list[str]:
    return [m.group(1).replace("\\|", "|").split("|")[0] for m in WIKILINK.finditer(text)]


def frontmatter_links(fm: dict) -> list[str]:
    out = []
    for v in fm.values():
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, str):
                out += wikilink_targets(item)
    return out


def test_vault_has_notes():
    assert len(VARIANT_NOTES) >= 30
    assert (NOTES / "00-index" / "RAG map.md").exists()


@pytest.mark.parametrize("path", CONTENT_NOTES, ids=rel)
def test_frontmatter_is_valid(path):
    fm = read_frontmatter(path)
    assert fm is not None, "missing frontmatter"
    assert fm.get("type") in TYPES


@pytest.mark.parametrize("path", VARIANT_NOTES, ids=rel)
def test_variant_schema(path):
    fm = read_frontmatter(path)
    assert fm["type"] in {"rag-variant", "rag-concept"}
    assert list(fm) == VARIANT_FIELDS
    assert fm["family"] in FAMILIES and path.parent.name == fm["family"]
    for key in ["adds_latency", "ingest_cost", "query_cost"]:
        assert fm[key] in LEVELS, key
    assert fm["status"] in STATUSES
    for key in ["aliases", "requires", "combines_with", "alternatives", "code", "papers", "tags"]:
        assert isinstance(fm[key], list), key
    for key in ["requires", "combines_with", "alternatives"]:
        for item in fm[key]:
            assert WIKILINK.fullmatch(item), f"{key}: {item!r} is not a [[wikilink]]"
            assert item != f"[[{path.stem}]]", f"{key} links to itself"
    for code in fm["code"]:
        assert (ROOT / code).exists(), f"code path missing: {code}"
    if fm["status"] != "idea":
        assert fm["code"], "implemented notes must list their code"


@pytest.mark.parametrize("path", CONTENT_NOTES, ids=rel)
def test_wikilinks_resolve(path):
    targets = wikilink_targets(prose(path)) + frontmatter_links(read_frontmatter(path) or {})
    for target in targets:
        name, _, heading = target.partition("#")
        name = name or path.stem
        assert name in NOTE_NAMES, f"unresolved [[{target}]]"
        if heading:
            assert heading in headings(NOTE_NAMES[name]), f"no heading '{heading}' in {name}"


@pytest.mark.parametrize("path", CONTENT_NOTES, ids=rel)
def test_relative_links_exist(path):
    for target in MD_LINK.findall(prose(path)):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        assert (path.parent / target.split("#")[0]).resolve().exists(), f"broken link: {target}"


def dataview_queries():
    for path in ALL_NOTES:
        for m in DATAVIEW.finditer(path.read_text(encoding="utf-8")):
            yield pytest.param(path, m.group(1), id=f"{rel(path)}#{m.start()}")


@pytest.mark.parametrize("path,query", list(dataview_queries()))
def test_dataview_fields_exist(path, query):
    m = re.match(r"TABLE\s+(?:WITHOUT ID\s+)?(.+?)\s*\nFROM\s+\"([^\"]+)\"", query, re.S)
    assert m, "expected TABLE ... FROM \"...\""
    fields = [re.split(r"\s+AS\s+", f.strip())[0] for f in m.group(1).split(",")]
    source = m.group(2)
    if source.startswith("notes/results"):
        allowed = set(RESULT_FIELDS)
    elif source.startswith("notes/variants"):
        allowed = set(VARIANT_FIELDS)
    else:
        pytest.fail(f"unexpected FROM {source}")
    sort_fields = re.findall(r"^SORT\s+(.+)$", query, re.M)
    for clause in sort_fields:
        fields += [s.split()[0] for s in clause.split(",")]
    for f in fields:
        assert f in allowed or DATAVIEW_IMPLICIT.match(f), f"unknown field {f!r} for {source}"


@pytest.mark.parametrize("key", list(VARIANTS))
def test_benchmarked_variants_have_notes_with_results_query(key):
    v = VARIANTS[key]
    assert v.note in NOTE_NAMES, f"no note named {v.note}"
    assert f'FROM "notes/results/{v.slug}"' in NOTE_NAMES[v.note].read_text(encoding="utf-8")


def test_rag_map_has_mermaid_flowchart():
    text = (NOTES / "00-index" / "RAG map.md").read_text(encoding="utf-8")
    assert re.search(r"^```mermaid\nflowchart TD\n", text, re.M)
