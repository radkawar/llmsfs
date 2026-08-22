import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from tokenizers import Tokenizer

from config import TransformerConfig, get_config, get_tokenizer_file_path, latest_weights_file_path
from dataset import causal_mask, special_token_id
from model import Transformer, build_transformer


@dataclass(frozen=True)
class InferenceBundle:
    config: TransformerConfig
    model: Transformer
    tokenizer_src: Tokenizer
    tokenizer_tgt: Tokenizer
    device: torch.device
    checkpoint_path: Path


def select_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None and str(preferred) != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_inference_bundle(
    config: TransformerConfig | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> InferenceBundle:
    """Load tokenizers, construct the model, and restore a checkpoint."""
    config = get_config() if config is None else config
    resolved_device = select_device(device)

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
) -> torch.Tensor:
    """Generate one target token at a time using the highest-logit token."""
    sos_id = special_token_id(tokenizer_tgt, "[SOS]")
    eos_id = special_token_id(tokenizer_tgt, "[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_id, dtype=source.dtype, device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1), device=device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        next_token_id = int(model.project(decoder_output[:, -1]).argmax(dim=-1).item())
        next_token = torch.tensor([[next_token_id]], dtype=source.dtype, device=device)
        decoder_input = torch.cat((decoder_input, next_token), dim=1)

        if next_token_id == eos_id:
            break

    return decoder_input.squeeze(0)


def translate(sentence: str, bundle: InferenceBundle | None = None) -> str:
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
    output_ids = greedy_decode(
        bundle.model,
        source,
        source_mask,
        bundle.tokenizer_tgt,
        bundle.config["seq_len"],
        bundle.device,
    )
    return bundle.tokenizer_tgt.decode(output_ids.detach().cpu().tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English text with the latest trained checkpoint.")
    parser.add_argument("sentence", nargs="?", default="I am a student.")
    parser.add_argument("--checkpoint", type=Path, help="Use a specific .pt checkpoint instead of the latest one.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    bundle = load_inference_bundle(checkpoint_path=args.checkpoint, device=args.device)
    print(f"Device:     {bundle.device}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Source:     {args.sentence}")
    print(f"Translation:{translate(args.sentence, bundle)}")


if __name__ == "__main__":
    main()
