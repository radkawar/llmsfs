import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from runtime import configure_runtime, resolve_amp_dtype, select_device

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import TransformerConfig, get_config, get_tokenizer_file_path, latest_weights_file_path
from dataset import special_token_id
from model import AttentionKVCache, DecoderLayerCache, Transformer, build_transformer


@dataclass(frozen=True)
class InferenceBundle:
    config: TransformerConfig
    model: Transformer
    tokenizer_src: Tokenizer
    tokenizer_tgt: Tokenizer
    device: torch.device
    amp_dtype: torch.dtype | None
    checkpoint_path: Path


def clean_decoded_text(text: str) -> str:
    """Undo spacing artifacts introduced by the simple word-level decoder."""
    text = re.sub(r"\s+([,.;:!?%…\)\]\}»])", r"\1", text)
    text = re.sub(r"([«\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s*([’'])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_token_ids(tokenizer: Tokenizer, token_ids: torch.Tensor | list[int]) -> str:
    """Decode target IDs and apply punctuation-aware spacing cleanup."""
    ids = token_ids.detach().cpu().tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    return clean_decoded_text(tokenizer.decode(ids, skip_special_tokens=True))


def load_inference_bundle(
    config: TransformerConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> InferenceBundle:
    """Load tokenizers, construct the model, and restore a checkpoint."""
    config = get_config() if config is None else config
    resolved_device = select_device(device)
    configure_runtime(resolved_device, config["seed"])
    amp_dtype = resolve_amp_dtype(resolved_device, config["precision"])

    src_tokenizer_path = get_tokenizer_file_path(config, config["lang_src"])
    tgt_tokenizer_path = get_tokenizer_file_path(config, config["lang_tgt"])
    missing_tokenizers = [path for path in (src_tokenizer_path, tgt_tokenizer_path) if not path.exists()]
    if missing_tokenizers:
        missing = ", ".join(str(path) for path in missing_tokenizers)
        raise FileNotFoundError(f"Missing tokenizer file(s): {missing}. Run training first.")

    tokenizer_src = Tokenizer.from_file(str(src_tokenizer_path))
    tokenizer_tgt = Tokenizer.from_file(str(tgt_tokenizer_path))

    if checkpoint_path is None:
        resolved_checkpoint = latest_weights_file_path(config)
        if resolved_checkpoint is None:
            raise FileNotFoundError("No model checkpoint was found. Run training first.")
    else:
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        use_fused_attention=config["use_fused_attention"],
        tie_target_embeddings=config["tie_target_embeddings"],
    ).to(resolved_device)

    checkpoint = torch.load(resolved_checkpoint, map_location=resolved_device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has an unexpected format: {resolved_checkpoint}")
    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model_state_dict"]))
    model.eval()

    return InferenceBundle(
        config=config,
        model=model,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        device=resolved_device,
        amp_dtype=amp_dtype,
        checkpoint_path=resolved_checkpoint,
    )


def encode_source_sentence(
    sentence: str,
    tokenizer_src: Tokenizer,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer_src.encode(sentence).ids
    padding = max_len - len(token_ids) - 2
    if padding < 0:
        raise ValueError(f"Input has {len(token_ids)} tokens, but at most {max_len - 2} are allowed")

    source = torch.tensor(
        [
            special_token_id(tokenizer_src, "[SOS]"),
            *token_ids,
            special_token_id(tokenizer_src, "[EOS]"),
            *([special_token_id(tokenizer_src, "[PAD]")] * padding),
        ],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    source_mask = (source != special_token_id(tokenizer_src, "[PAD]")).unsqueeze(1).unsqueeze(1)
    return source, source_mask


@torch.inference_mode()
def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Generate highest-scoring tokens incrementally with an inference KV cache."""
    sos_id = special_token_id(tokenizer_tgt, "[SOS]")
    eos_id = special_token_id(tokenizer_tgt, "[EOS]")
    _validate_decoding_options(max_len, 1, repetition_penalty, no_repeat_ngram_size)

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.full((1, 1), sos_id, dtype=source.dtype, device=device)
        current_token = decoder_input
        cache: list[DecoderLayerCache] | None = None

        for position in range(max_len - 1):
            decoder_output, cache = model.decode_step(
                encoder_output,
                source_mask,
                current_token,
                position,
                cache,
            )
            logits = model.project(decoder_output[:, -1])
            logits = _apply_generation_constraints(
                logits,
                decoder_input,
                eos_id,
                repetition_penalty,
                no_repeat_ngram_size,
            )
            next_token_id = int(logits.argmax(dim=-1).item())
            next_token = torch.tensor([[next_token_id]], dtype=source.dtype, device=device)
            decoder_input = torch.cat((decoder_input, next_token), dim=1)
            current_token = next_token

            if next_token_id == eos_id:
                break

    return decoder_input.squeeze(0)


def _validate_decoding_options(
    max_len: int,
    beam_size: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative")


def _apply_generation_constraints(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    eos_id: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    if repetition_penalty == 1.0 and no_repeat_ngram_size == 0:
        return logits

    constrained = logits.clone()
    if repetition_penalty > 1.0:
        seen_logits = constrained.gather(1, sequences)
        penalized_logits = torch.where(
            seen_logits < 0,
            seen_logits * repetition_penalty,
            seen_logits / repetition_penalty,
        )
        constrained.scatter_(1, sequences, penalized_logits)

    if no_repeat_ngram_size == 1:
        for row in range(sequences.size(0)):
            blocked_tokens = sequences[row]
            blocked_tokens = blocked_tokens[blocked_tokens != eos_id]
            constrained[row, blocked_tokens] = torch.finfo(constrained.dtype).min
    elif no_repeat_ngram_size > 1 and sequences.size(1) >= no_repeat_ngram_size:
        ngrams = sequences.unfold(1, no_repeat_ngram_size, 1)
        current_prefix = sequences[:, -(no_repeat_ngram_size - 1) :]
        matching_prefixes = (ngrams[:, :, :-1] == current_prefix.unsqueeze(1)).all(dim=-1)
        following_tokens = ngrams[:, :, -1]
        for row in range(sequences.size(0)):
            blocked_tokens = following_tokens[row][matching_prefixes[row]]
            blocked_tokens = blocked_tokens[blocked_tokens != eos_id]
            constrained[row, blocked_tokens] = torch.finfo(constrained.dtype).min
    return constrained


def _reorder_cache(cache: list[DecoderLayerCache], beam_indices: torch.Tensor) -> list[DecoderLayerCache]:
    reordered: list[DecoderLayerCache] = []
    for layer_cache in cache:
        self_attention = layer_cache.self_attention
        cross_attention = layer_cache.cross_attention
        if self_attention is None or cross_attention is None:
            raise ValueError("Cannot reorder an incomplete decoder cache")
        reordered.append(
            DecoderLayerCache(
                self_attention=AttentionKVCache(
                    self_attention.key.index_select(0, beam_indices),
                    self_attention.value.index_select(0, beam_indices),
                ),
                cross_attention=AttentionKVCache(
                    cross_attention.key.index_select(0, beam_indices),
                    cross_attention.value.index_select(0, beam_indices),
                ),
            )
        )
    return reordered


@torch.inference_mode()
def beam_search_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Keep the best partial translations while incrementally reusing KV state."""
    _validate_decoding_options(max_len, beam_size, repetition_penalty, no_repeat_ngram_size)
    if length_penalty < 0:
        raise ValueError("length_penalty cannot be negative")
    if source.size(0) != 1:
        raise ValueError("Beam search currently expects one source sentence")
    if beam_size == 1:
        return greedy_decode(
            model,
            source,
            source_mask,
            tokenizer_tgt,
            max_len,
            device,
            amp_dtype,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    sos_id = special_token_id(tokenizer_tgt, "[SOS]")
    eos_id = special_token_id(tokenizer_tgt, "[EOS]")
    vocab_size = tokenizer_tgt.get_vocab_size()
    if beam_size > vocab_size:
        raise ValueError(f"beam_size {beam_size} exceeds target vocabulary size {vocab_size}")

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        encoder_output = model.encode(source, source_mask).expand(beam_size, -1, -1).contiguous()
        expanded_source_mask = source_mask.expand(beam_size, -1, -1, -1)
        generated = torch.full((beam_size, 1), sos_id, dtype=source.dtype, device=device)
        current_token = generated
        beam_scores = torch.full((beam_size,), torch.finfo(torch.float32).min, device=device)
        beam_scores[0] = 0.0
        finished = torch.zeros(beam_size, dtype=torch.bool, device=device)
        sequence_lengths = torch.zeros(beam_size, dtype=torch.long, device=device)
        cache: list[DecoderLayerCache] | None = None

        for position in range(max_len - 1):
            decoder_output, cache = model.decode_step(
                encoder_output,
                expanded_source_mask,
                current_token,
                position,
                cache,
            )
            logits = model.project(decoder_output[:, -1])
            logits = _apply_generation_constraints(
                logits,
                generated,
                eos_id,
                repetition_penalty,
                no_repeat_ngram_size,
            )
            log_probabilities = F.log_softmax(logits.float(), dim=-1)

            forced_eos = torch.full_like(log_probabilities, torch.finfo(log_probabilities.dtype).min)
            forced_eos[:, eos_id] = 0.0
            log_probabilities = torch.where(finished.unsqueeze(1), forced_eos, log_probabilities)

            candidate_scores = beam_scores.unsqueeze(1) + log_probabilities
            beam_scores, candidate_indices = candidate_scores.reshape(-1).topk(beam_size)
            parent_beams = candidate_indices // vocab_size
            next_tokens = candidate_indices % vocab_size

            generated = torch.cat((generated.index_select(0, parent_beams), next_tokens.unsqueeze(1)), dim=1)
            parent_finished = finished.index_select(0, parent_beams)
            sequence_lengths = sequence_lengths.index_select(0, parent_beams)
            newly_finished = (~parent_finished) & (next_tokens == eos_id)
            sequence_lengths = torch.where(
                newly_finished,
                torch.full_like(sequence_lengths, position + 1),
                sequence_lengths,
            )
            finished = parent_finished | newly_finished
            cache = _reorder_cache(cache, parent_beams)
            current_token = next_tokens.unsqueeze(1)

            if bool(finished.all().item()):
                break

        sequence_lengths = torch.where(
            sequence_lengths == 0,
            torch.full_like(sequence_lengths, generated.size(1) - 1),
            sequence_lengths,
        )
        length_normalizer = ((5.0 + sequence_lengths.float()) / 6.0).pow(length_penalty)
        best_beam = int((beam_scores / length_normalizer).argmax().item())
        best_sequence = generated[best_beam]

    eos_positions = (best_sequence == eos_id).nonzero(as_tuple=False)
    if eos_positions.numel() > 0:
        best_sequence = best_sequence[: int(eos_positions[0].item()) + 1]
    return best_sequence


def decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    """Decode with greedy search (beam 1) or cached beam search (beam > 1)."""
    if beam_size == 1:
        return greedy_decode(
            model,
            source,
            source_mask,
            tokenizer_tgt,
            max_len,
            device,
            amp_dtype,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return beam_search_decode(
        model,
        source,
        source_mask,
        tokenizer_tgt,
        max_len,
        device,
        amp_dtype,
        beam_size=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


def translate(
    sentence: str,
    bundle: InferenceBundle | None = None,
    *,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    max_len: int | None = None,
) -> str:
    """Translate one English sentence with a loaded or automatically loaded model."""
    if not sentence.strip():
        raise ValueError("The sentence cannot be empty")

    bundle = load_inference_bundle() if bundle is None else bundle
    source, source_mask = encode_source_sentence(
        sentence,
        bundle.tokenizer_src,
        bundle.config["seq_len"],
        bundle.device,
    )
    generation_limit = min(max_len or bundle.config["validation_max_len"], bundle.config["seq_len"])
    output_ids = decode(
        bundle.model,
        source,
        source_mask,
        bundle.tokenizer_tgt,
        generation_limit,
        bundle.device,
        bundle.amp_dtype,
        beam_size=beam_size,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return decode_token_ids(bundle.tokenizer_tgt, output_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English text with the latest trained checkpoint.")
    parser.add_argument("sentence", nargs="?", default="I am a student.")
    parser.add_argument("--checkpoint", type=Path, help="Use a specific .pt checkpoint instead of the latest one.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--beam-size", type=int, default=3, help="Use 1 for greedy decoding; 3-5 is a practical beam width.")
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="Use 0 to disable n-gram blocking.")
    parser.add_argument("--max-length", type=int, help="Maximum output tokens, including [SOS].")
    args = parser.parse_args()

    bundle = load_inference_bundle(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Device:     {bundle.device}")
    print(f"Precision:  {bundle.amp_dtype or torch.float32}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Search:     {'greedy' if args.beam_size == 1 else f'beam {args.beam_size}'}")
    print(f"Source:     {args.sentence}")
    print(
        f"Translation: {translate(args.sentence, bundle, beam_size=args.beam_size, length_penalty=args.length_penalty, repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size, max_len=args.max_length)}"
    )


if __name__ == "__main__":
    main()
