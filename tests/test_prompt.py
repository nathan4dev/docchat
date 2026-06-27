from core.prompt import build_context_block, build_user_prompt


def test_build_context_block_numbers_chunks_from_one():
    chunks = [{"text": "first", "source": "a.txt"}, {"text": "second", "source": "b.txt"}]
    block = build_context_block(chunks)
    assert "[1] (source: a.txt)\nfirst" in block
    assert "[2] (source: b.txt)\nsecond" in block


def test_build_context_block_empty_list():
    assert build_context_block([]) == ""


def test_build_user_prompt_includes_question_and_context():
    chunks = [{"text": "the sky is blue", "source": "facts.txt"}]
    prompt = build_user_prompt("What color is the sky?", chunks)
    assert "What color is the sky?" in prompt
    assert "the sky is blue" in prompt
    assert "Context:" in prompt
