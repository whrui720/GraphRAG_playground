from rag.standard_rag import TextChunker


def test_windows_overlap_by_configured_amount():
    text = "".join(chr(ord("a") + i % 26) for i in range(1200))
    chunks = TextChunker(chunk_size=500, overlap=100).chunk_text(text, source="x.txt")

    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "x.txt" for c in chunks)
    assert len(chunks[0].text) == 500
    # consecutive windows start 400 chars apart, so they share 100 chars
    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_chunk_count_is_ceil_length_over_step():
    # step = 500, so 1200 chars -> windows starting at 0, 500, 1000
    chunks = TextChunker(chunk_size=600, overlap=100).chunk_text("x" * 1200)
    assert len(chunks) == 3


def test_tail_window_can_be_contained_in_previous_one():
    # Known quirk (documented in the Chunking strategies note): the loop runs
    # while i < len(text), so the last window can be a strict suffix of the
    # previous one and adds no new text.
    text = "".join(str(i % 10) for i in range(1050))
    chunks = TextChunker(chunk_size=600, overlap=100).chunk_text(text)
    assert len(chunks) == 3
    assert chunks[1].text.endswith(chunks[2].text)


def test_whitespace_only_windows_are_dropped():
    assert TextChunker(chunk_size=10, overlap=0).chunk_text(" " * 30) == []
