from core.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", "doc.txt") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("hello world", "doc.txt", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].source == "doc.txt"


def test_long_text_is_split_with_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = chunk_text(text, "doc.txt", chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # consecutive chunks should share their overlap region
    first_tail = chunks[0].text[-50:]
    second_head = chunks[1].text[:50]
    assert first_tail == second_head


def test_chunk_indexes_are_sequential():
    text = "x" * 1000
    chunks = chunk_text(text, "doc.txt", chunk_size=300, overlap=50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("text", "doc.txt", chunk_size=100, overlap=100)
        assert False, "expected ValueError"
    except ValueError:
        pass
