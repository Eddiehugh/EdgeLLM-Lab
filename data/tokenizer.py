"""Tokenizer wrappers and registry."""

from __future__ import annotations

from core.registry import Registry, build_from_config


TOKENIZER_REGISTRY = Registry("tokenizer")


class TokenizerWrapper:
    """Minimal tokenizer interface used by the framework."""

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError


@TOKENIZER_REGISTRY.register("char")
class CharTokenizer(TokenizerWrapper):
    """Small character tokenizer for smoke tests and tiny experiments."""

    def __init__(self, vocab: list[str] | None = None):
        vocab = vocab or [chr(i) for i in range(256)]
        self.id_to_token = list(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.unk_token_id = self.token_to_id.get("?", 0)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(char, self.unk_token_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in token_ids)


def build_tokenizer(tokenizer_type: str | dict = "char", **kwargs) -> TokenizerWrapper:
    """Build a tokenizer by name."""
    return build_from_config(
        TOKENIZER_REGISTRY,
        tokenizer_type,
        default_type="char",
        **kwargs,
    )
