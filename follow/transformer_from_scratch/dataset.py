from collections.abc import Mapping
from typing import Protocol, TypedDict, cast

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class TranslationRows(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


class BilingualSample(TypedDict):
    encoder_input: torch.Tensor
    decoder_input: torch.Tensor
    encoder_mask: torch.Tensor
    decoder_mask: torch.Tensor
    label: torch.Tensor
    src_text: str
    tgt_text: str


class BilingualBatch(TypedDict):
    encoder_input: torch.Tensor
    decoder_input: torch.Tensor
    encoder_mask: torch.Tensor
    decoder_mask: torch.Tensor
    label: torch.Tensor
    src_text: list[str]
    tgt_text: list[str]


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not contain the required token {token!r}")
    return token_id


class BilingualDataset(Dataset[BilingualSample]):
    def __init__(
        self,
        ds: TranslationRows,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.src_sos_id = special_token_id(tokenizer_src, "[SOS]")
        self.src_eos_id = special_token_id(tokenizer_src, "[EOS]")
        self.src_pad_id = special_token_id(tokenizer_src, "[PAD]")
        self.tgt_sos_id = special_token_id(tokenizer_tgt, "[SOS]")
        self.tgt_eos_id = special_token_id(tokenizer_tgt, "[EOS]")
        self.tgt_pad_id = special_token_id(tokenizer_tgt, "[PAD]")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> BilingualSample:
        row = self.ds[index]
        translations = cast(Mapping[str, str], row["translation"])
        src_text = translations[self.src_lang]
        tgt_text = translations[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # The encoder receives [SOS] sentence [EOS]. The decoder receives
        # [SOS] sentence, while its label is sentence [EOS].
        enc_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_padding = self.seq_len - len(dec_input_tokens) - 1
        if enc_padding < 0 or dec_padding < 0:
            raise ValueError(
                f"Sentence pair is too long for seq_len={self.seq_len}: "
                f"source={len(enc_input_tokens)} tokens, target={len(dec_input_tokens)} tokens"
            )

        encoder_input = torch.tensor(
            [self.src_sos_id, *enc_input_tokens, self.src_eos_id, *([self.src_pad_id] * enc_padding)],
            dtype=torch.long,
        )
        decoder_input = torch.tensor(
            [self.tgt_sos_id, *dec_input_tokens, *([self.tgt_pad_id] * dec_padding)],
            dtype=torch.long,
        )
        label = torch.tensor(
            [*dec_input_tokens, self.tgt_eos_id, *([self.tgt_pad_id] * dec_padding)],
            dtype=torch.long,
        )

        if not (encoder_input.size(0) == decoder_input.size(0) == label.size(0) == self.seq_len):
            raise RuntimeError("Token construction produced an unexpected sequence length")

        encoder_mask = (encoder_input != self.src_pad_id).unsqueeze(0).unsqueeze(0)
        decoder_padding_mask = (decoder_input != self.tgt_pad_id).unsqueeze(0)
        decoder_mask = decoder_padding_mask & causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular mask with shape ``(1, size, size)``."""
    return torch.ones((1, size, size), dtype=torch.bool, device=device).tril()
