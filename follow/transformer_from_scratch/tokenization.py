from collections.abc import Iterable
from typing import Literal, Protocol

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


TokenizerType = Literal["wordlevel", "byte_bpe"]
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


class TokenizerBuilder(Protocol):
    """Common interface for tokenizer training strategies."""

    def build(self, sentences: Iterable[str], vocab_size: int, min_frequency: int) -> Tokenizer: ...


class WordLevelTokenizerBuilder:
    def build(self, sentences: Iterable[str], vocab_size: int, min_frequency: int) -> Tokenizer:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        return tokenizer


class ByteBPETokenizerBuilder:
    def build(self, sentences: Iterable[str], vocab_size: int, min_frequency: int) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = NFC()
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        return tokenizer


TOKENIZER_BUILDERS: dict[TokenizerType, TokenizerBuilder] = {
    "wordlevel": WordLevelTokenizerBuilder(),
    "byte_bpe": ByteBPETokenizerBuilder(),
}


def build_tokenizer(
    tokenizer_type: TokenizerType,
    sentences: Iterable[str],
    vocab_size: int,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a tokenizer through the selected builder interface."""
    if vocab_size < len(SPECIAL_TOKENS) + 256 and tokenizer_type == "byte_bpe":
        raise ValueError("byte_bpe vocab_size must leave room for the 256-byte alphabet and special tokens")
    return TOKENIZER_BUILDERS[tokenizer_type].build(sentences, vocab_size, min_frequency)
