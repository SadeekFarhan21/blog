"""A deliberately simple character-level tokenizer."""

from __future__ import annotations


class CharacterTokenizer:
    """Map every unique character in a corpus to a stable integer ID."""

    def __init__(self, vocabulary: str) -> None:
        characters = sorted(set(vocabulary))
        if not characters:
            raise ValueError("vocabulary must contain at least one character")

        self._characters = tuple(characters)
        self._token_to_id = {character: index for index, character in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self._characters)

    def encode(self, text: str) -> list[int]:
        try:
            return [self._token_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is not in the vocabulary") from error

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self._characters[token_id] for token_id in token_ids)
        except IndexError as error:
            raise ValueError("token ID is outside the vocabulary") from error

