import pytest

from tokenizer import CharacterTokenizer


def test_character_tokenizer_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("hello world")
    text = "hello"

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == len(set("hello world"))


def test_character_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="not in the vocabulary"):
        tokenizer.encode("abcd")
