import math
import random
from collections.abc import Iterator, Mapping
from typing import Protocol, TypedDict, cast

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, Sampler


class TranslationRows(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


class BilingualSample(TypedDict):
    src_ids: list[int]
    tgt_ids: list[int]
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
    source_token_count: int
    target_token_count: int


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not contain the required token {token!r}")
    return token_id


class BilingualDataset(Dataset[BilingualSample]):
    """A translation dataset that tokenizes each sentence only once."""

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
        self.seq_len = seq_len

        src_texts: list[str] = []
        tgt_texts: list[str] = []
        for index in range(len(ds)):
            row = ds[index]
            translations = cast(Mapping[str, str], row["translation"])
            src_texts.append(translations[src_lang])
            tgt_texts.append(translations[tgt_lang])

        # encode_batch uses the Rust tokenizer in one call. Keeping the IDs in
        # memory avoids repeating tokenization during every training epoch.
        src_encodings = tokenizer_src.encode_batch(src_texts)
        tgt_encodings = tokenizer_tgt.encode_batch(tgt_texts)
        self.samples: list[BilingualSample] = []
        self.lengths: list[int] = []
        self.max_src_len = 0
        self.max_tgt_len = 0
        self.skipped_count = 0

        for src_text, tgt_text, src_encoding, tgt_encoding in zip(
            src_texts,
            tgt_texts,
            src_encodings,
            tgt_encodings,
            strict=True,
        ):
            src_ids = src_encoding.ids
            tgt_ids = tgt_encoding.ids
            src_len = len(src_ids) + 2  # [SOS] source [EOS]
            tgt_len = len(tgt_ids) + 1  # [SOS] target / target [EOS]
            if src_len > seq_len or tgt_len > seq_len:
                self.skipped_count += 1
                continue

            self.samples.append(
                {
                    "src_ids": src_ids,
                    "tgt_ids": tgt_ids,
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                }
            )
            self.lengths.append(max(src_len, tgt_len))
            self.max_src_len = max(self.max_src_len, src_len)
            self.max_tgt_len = max(self.max_tgt_len, tgt_len)

        if not self.samples:
            raise ValueError(f"No sentence pairs fit within seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BilingualSample:
        return self.samples[index]

    def sequence_length(self, index: int) -> int:
        return self.lengths[index]


class BilingualCollator:
    """Dynamically pad a batch to its longest source and target sentences."""

    def __init__(
        self,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        seq_len: int,
        pad_to_multiple_of: int = 8,
    ) -> None:
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")

        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.src_sos_id = special_token_id(tokenizer_src, "[SOS]")
        self.src_eos_id = special_token_id(tokenizer_src, "[EOS]")
        self.src_pad_id = special_token_id(tokenizer_src, "[PAD]")
        self.tgt_sos_id = special_token_id(tokenizer_tgt, "[SOS]")
        self.tgt_eos_id = special_token_id(tokenizer_tgt, "[EOS]")
        self.tgt_pad_id = special_token_id(tokenizer_tgt, "[PAD]")

    def _padded_length(self, length: int) -> int:
        rounded = math.ceil(length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        return min(max(length, rounded), self.seq_len)

    def __call__(self, samples: list[BilingualSample]) -> BilingualBatch:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        source_lengths = [len(sample["src_ids"]) + 2 for sample in samples]
        target_lengths = [len(sample["tgt_ids"]) + 1 for sample in samples]
        max_source_length = self._padded_length(max(source_lengths))
        max_target_length = self._padded_length(max(target_lengths))
        batch_size = len(samples)

        encoder_input = torch.full((batch_size, max_source_length), self.src_pad_id, dtype=torch.long)
        decoder_input = torch.full((batch_size, max_target_length), self.tgt_pad_id, dtype=torch.long)
        label = torch.full((batch_size, max_target_length), self.tgt_pad_id, dtype=torch.long)

        for row, sample in enumerate(samples):
            src_ids = sample["src_ids"]
            tgt_ids = sample["tgt_ids"]
            encoder_tokens = torch.tensor([self.src_sos_id, *src_ids, self.src_eos_id], dtype=torch.long)
            decoder_tokens = torch.tensor([self.tgt_sos_id, *tgt_ids], dtype=torch.long)
            label_tokens = torch.tensor([*tgt_ids, self.tgt_eos_id], dtype=torch.long)
            encoder_input[row, : encoder_tokens.numel()] = encoder_tokens
            decoder_input[row, : decoder_tokens.numel()] = decoder_tokens
            label[row, : label_tokens.numel()] = label_tokens

        encoder_mask = (encoder_input != self.src_pad_id).unsqueeze(1).unsqueeze(1)
        decoder_padding_mask = (decoder_input != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
        decoder_mask = decoder_padding_mask & causal_mask(max_target_length).unsqueeze(1)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": [sample["src_text"] for sample in samples],
            "tgt_text": [sample["tgt_text"] for sample in samples],
            "source_token_count": sum(source_lengths),
            "target_token_count": sum(target_lengths),
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle batches while grouping similarly sized sentences together."""

    def __init__(
        self,
        dataset: BilingualDataset,
        batch_size: int,
        *,
        bucket_size_multiplier: int = 10,
        seed: int = 1337,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if bucket_size_multiplier < 1:
            raise ValueError("bucket_size_multiplier must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        sorted_indices = sorted(range(len(self.dataset)), key=self.dataset.sequence_length)
        batches: list[list[int]] = []

        for start in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[start : start + self.bucket_size]
            rng.shuffle(bucket)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


class SortedBatchSampler(Sampler[list[int]]):
    """Visit every example once in length-sorted batches for efficient validation."""

    def __init__(self, dataset: BilingualDataset, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        sorted_indices = sorted(range(len(self.dataset)), key=self.dataset.sequence_length)
        for start in range(0, len(sorted_indices), self.batch_size):
            yield sorted_indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


def representative_indices(dataset: BilingualDataset, examples_per_bucket: int) -> list[tuple[str, int]]:
    """Choose deterministic examples from the short, medium, and long thirds."""
    if examples_per_bucket < 1:
        raise ValueError("examples_per_bucket must be positive")
    if len(dataset) < 3:
        raise ValueError("At least three validation examples are required")

    sorted_indices = sorted(range(len(dataset)), key=dataset.sequence_length)
    boundaries = (0, len(sorted_indices) // 3, 2 * len(sorted_indices) // 3, len(sorted_indices))
    selected: list[tuple[str, int]] = []

    for bucket_index, label in enumerate(("short", "medium", "long")):
        bucket = sorted_indices[boundaries[bucket_index] : boundaries[bucket_index + 1]]
        count = min(examples_per_bucket, len(bucket))
        for sample_index in range(count):
            # Interior quantiles avoid selecting only the most extreme lengths.
            position = ((sample_index + 1) * len(bucket)) // (count + 1)
            selected.append((label, bucket[position]))
    return selected


def causal_mask(size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular mask with shape ``(1, size, size)``."""
    return torch.ones((1, size, size), dtype=torch.bool, device=device).tril()
