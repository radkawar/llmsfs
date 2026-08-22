import argparse
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from typing import TypedDict, cast

from runtime import configure_runtime, resolve_amp_dtype, select_device

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tqdm.auto import tqdm

from config import (
    TransformerConfig,
    get_experiment_path,
    get_tokenizer_file_path,
    get_weights_file_path,
    get_weights_folder_path,
    latest_weights_file_path,
    load_config,
)
from dataset import (
    BilingualBatch,
    BilingualCollator,
    BilingualDataset,
    LengthBucketBatchSampler,
    SortedBatchSampler,
    TranslationRows,
    representative_indices,
    special_token_id,
)
from model import Transformer, build_transformer
from tokenization import build_tokenizer
from translate import decode, decode_token_ids


class ValidationMetrics(TypedDict):
    cer: float
    wer: float
    bleu: float


def evaluate_validation_loss(
    model: Transformer,
    validation_ds: DataLoader[BilingualBatch],
    tokenizer_tgt: Tokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    print_msg: Callable[[str], None],
    global_step: int,
    writer: SummaryWriter | None,
) -> float:
    """Measure teacher-forced, token-weighted loss over the full validation split."""
    model.eval()
    pad_id = special_token_id(tokenizer_tgt, "[PAD]")
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_tokens = 0

    with torch.inference_mode():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
                summed_loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer_tgt.get_vocab_size()),
                    label.reshape(-1),
                    ignore_index=pad_id,
                    label_smoothing=0.1,
                    reduction="sum",
                )
            total_loss.add_(summed_loss.float())
            total_tokens += batch["target_token_count"]

    if total_tokens == 0:
        raise ValueError("The validation split contained no target tokens")
    average_loss = (total_loss / total_tokens).item()
    print_msg(f"Full teacher-forced validation | loss {average_loss:.3f} | {total_tokens:,} target tokens")
    if writer is not None:
        writer.add_scalar("validation/loss", average_loss, global_step)
    return average_loss


def run_validation(
    model: Transformer,
    validation_ds: DataLoader[BilingualBatch],
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    print_msg: Callable[[str], None],
    global_step: int,
    writer: SummaryWriter | None,
    length_buckets: list[str],
    beam_size: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> ValidationMetrics:
    model.eval()
    expected: list[str] = []
    predicted: list[str] = []
    console_width = shutil.get_terminal_size(fallback=(100, 24)).columns

    with torch.inference_mode():
        for count, batch in enumerate(validation_ds, start=1):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            if encoder_input.size(0) != 1:
                raise ValueError("Validation requires a batch size of 1")

            model_out = decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_tgt,
                max_len,
                device,
                amp_dtype,
                beam_size=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = decode_token_ids(tokenizer_tgt, model_out)
            expected.append(target_text)
            predicted.append(model_out_text)
            bucket = length_buckets[count - 1]
            source_tokens = int(encoder_mask.sum().item())
            target_tokens = batch["target_token_count"]

            print_msg("-" * console_width)
            print_msg(f"{bucket.upper()} SAMPLE | source {source_tokens} tokens | target {target_tokens} tokens")
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

        print_msg("-" * console_width)

    if not predicted:
        return {"cer": 0.0, "wer": 0.0, "bleu": 0.0}

    metrics: ValidationMetrics = {
        "cer": CharErrorRate()(predicted, expected).item(),
        "wer": WordErrorRate()(predicted, expected).item(),
        "bleu": BLEUScore()(predicted, expected).item(),
    }
    strategy = "greedy" if beam_size == 1 else f"beam {beam_size}"
    print_msg(
        f"Representative generation ({strategy}, {len(predicted)} examples) | "
        f"CER {metrics['cer']:.3f} | WER {metrics['wer']:.3f} | BLEU {metrics['bleu']:.3f}"
    )

    if writer is not None:
        writer.add_scalar("validation/cer", metrics["cer"], global_step)
        writer.add_scalar("validation/wer", metrics["wer"], global_step)
        writer.add_scalar("validation/bleu", metrics["bleu"], global_step)
        writer.flush()
    return metrics


def get_all_sentences(ds: HFDataset, language: str) -> Iterator[str]:
    for raw_item in ds:
        item = cast(Mapping[str, object], raw_item)
        translations = cast(Mapping[str, str], item["translation"])
        yield translations[language]


def get_or_build_tokenizer(config: TransformerConfig, ds: HFDataset, language: str) -> Tokenizer:
    tokenizer_path = get_tokenizer_file_path(config, language)
    if tokenizer_path.exists():
        print(f"Loading {config['tokenizer_type']} {language} tokenizer: {tokenizer_path}")
        return Tokenizer.from_file(str(tokenizer_path))

    print(f"Training {config['tokenizer_type']} {language} tokenizer (vocab target {config['tokenizer_vocab_size']:,})...")
    tokenizer = build_tokenizer(
        config["tokenizer_type"],
        get_all_sentences(ds, language),
        config["tokenizer_vocab_size"],
        config["tokenizer_min_frequency"],
    )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(
    config: TransformerConfig,
) -> tuple[
    DataLoader[BilingualBatch],
    DataLoader[BilingualBatch],
    DataLoader[BilingualBatch],
    list[str],
    Tokenizer,
    Tokenizer,
]:
    raw_dataset = load_dataset(
        config["datasource"],
        config["dataset_config"],
        split="train",
    )
    if not isinstance(raw_dataset, HFDataset):
        raise TypeError("Expected a regular Hugging Face Dataset")

    tokenizer_src = get_or_build_tokenizer(config, raw_dataset, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, raw_dataset, config["lang_tgt"])
    split = raw_dataset.train_test_split(test_size=0.1, seed=config["seed"])

    print("Pre-tokenizing the dataset once...")
    train_ds = BilingualDataset(
        cast(TranslationRows, split["train"]),
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        cast(TranslationRows, split["test"]),
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    collator = BilingualCollator(
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
        config["pad_to_multiple_of"],
    )
    batch_sampler = LengthBucketBatchSampler(
        train_ds,
        config["batch_size"],
        bucket_size_multiplier=config["bucket_size_multiplier"],
        seed=config["seed"],
    )
    train_dataloader = cast(
        DataLoader[BilingualBatch],
        DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collator),
    )
    validation_batch_sampler = SortedBatchSampler(val_ds, config["batch_size"])
    validation_loss_dataloader = cast(
        DataLoader[BilingualBatch],
        DataLoader(val_ds, batch_sampler=validation_batch_sampler, collate_fn=collator),
    )

    configured_indices = config["validation_example_indices"]
    if configured_indices:
        examples_per_bucket = config["validation_examples_per_bucket"]
        expected_count = 3 * examples_per_bucket
        if len(configured_indices) != expected_count:
            raise ValueError(f"Expected {expected_count} validation_example_indices, received {len(configured_indices)}")
        if any(index < 0 or index >= len(val_ds) for index in configured_indices):
            raise ValueError("A validation_example_indices entry is outside the validation split")
        selected_examples = [
            (("short", "medium", "long")[position // examples_per_bucket], index) for position, index in enumerate(configured_indices)
        ]
    else:
        selected_examples = representative_indices(val_ds, config["validation_examples_per_bucket"])
    length_buckets = [bucket for bucket, _ in selected_examples]
    representative_ds = Subset(val_ds, [index for _, index in selected_examples])
    validation_generation_dataloader = cast(
        DataLoader[BilingualBatch],
        DataLoader(representative_ds, batch_size=1, shuffle=False, collate_fn=collator),
    )

    max_src_len = max(train_ds.max_src_len, val_ds.max_src_len)
    max_tgt_len = max(train_ds.max_tgt_len, val_ds.max_tgt_len)
    print(f"Maximum source/target lengths: {max_src_len}/{max_tgt_len} tokens")
    print(f"Training/validation examples: {len(train_ds):,}/{len(val_ds):,}")
    if train_ds.skipped_count or val_ds.skipped_count:
        print(f"Skipped over-length training/validation pairs: {train_ds.skipped_count:,}/{val_ds.skipped_count:,}")
    print(f"Dynamic padding + length bucketing: {len(train_dataloader):,} batches (batch size {config['batch_size']})")
    print(
        f"Validation: {len(validation_loss_dataloader):,} teacher-forced batches + "
        f"{len(validation_generation_dataloader)} representative generations"
    )
    return (
        train_dataloader,
        validation_loss_dataloader,
        validation_generation_dataloader,
        length_buckets,
        tokenizer_src,
        tokenizer_tgt,
    )


def get_model(config: TransformerConfig, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        use_fused_attention=config["use_fused_attention"],
        tie_target_embeddings=config["tie_target_embeddings"],
    )


def _format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"
    return f"{minutes:d}m {remaining_seconds:02d}s"


def train_model(
    config: TransformerConfig,
    requested_device: str | torch.device | None = None,
    *,
    evaluate_only: bool = False,
) -> None:
    device = select_device(requested_device)
    configure_runtime(device, config["seed"])
    amp_dtype = resolve_amp_dtype(device, config["precision"])
    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"Device: {device}")
    print(f"Precision: {amp_dtype or torch.float32}")
    print(f"Experiment: {config['name']}")
    print(f"Tokenizer: {config['tokenizer_type']} (target vocab {config['tokenizer_vocab_size']:,})")
    attention_backend = "scaled-dot-product" if config["use_fused_attention"] else "MPS-fast explicit matmul"
    print(f"Optimizations: dynamic padding, length bucketing, cached tokenization, {attention_backend} attention, fused LayerNorm")
    get_weights_folder_path(config).mkdir(parents=True, exist_ok=True)

    (
        train_dataloader,
        validation_loss_dataloader,
        validation_generation_dataloader,
        validation_length_buckets,
        tokenizer_src,
        tokenizer_tgt,
    ) = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable_parameters / 1_000_000:.1f}M")

    writer = SummaryWriter(str(get_experiment_path(config)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        eps=1e-9,
        fused=device.type in {"mps", "cuda"},
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload is not None else None
    )
    if model_filename is not None:
        if not model_filename.is_file():
            raise FileNotFoundError(f"Requested preload checkpoint does not exist: {model_filename}")
        print(f"Resuming from {model_filename}")
        checkpoint = torch.load(model_filename, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint has an unexpected format: {model_filename}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        initial_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        # Loading optimizer state also loads its old learning rate. The current
        # config remains authoritative when a resumed run overrides --learning-rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config["lr"]
    else:
        print("No compatible checkpoint found; starting from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=special_token_id(tokenizer_tgt, "[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    if evaluate_only:
        try:
            evaluate_validation_loss(
                model,
                validation_loss_dataloader,
                tokenizer_tgt,
                device,
                amp_dtype,
                print,
                global_step,
                writer,
            )
            run_validation(
                model,
                validation_generation_dataloader,
                tokenizer_tgt,
                min(config["validation_max_len"], config["seq_len"]),
                device,
                amp_dtype,
                print,
                global_step,
                writer,
                validation_length_buckets,
                config["validation_beam_size"],
                config["validation_length_penalty"],
                config["validation_no_repeat_ngram_size"],
            )
        finally:
            writer.close()
        return

    try:
        for epoch in range(initial_epoch, config["num_epochs"]):
            model.train()
            epoch_start = time.perf_counter()
            epoch_loss = torch.zeros((), device=device)
            real_tokens = 0
            padded_tokens = 0
            examples_seen = 0
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch:02d}",
                dynamic_ncols=True,
                mininterval=1.0,
            )

            for step_in_epoch, batch in enumerate(batch_iterator, start=1):
                encoder_input = batch["encoder_input"].to(device, non_blocking=True)
                decoder_input = batch["decoder_input"].to(device, non_blocking=True)
                encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
                decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
                    loss = loss_fn(logits.reshape(-1, tokenizer_tgt.get_vocab_size()), label.reshape(-1))

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss.add_(loss.detach())
                examples_seen += encoder_input.size(0)
                real_tokens += batch["source_token_count"] + batch["target_token_count"]
                padded_tokens += encoder_input.numel() + decoder_input.numel()
                global_step += 1

                if step_in_epoch % config["log_every"] == 0 or step_in_epoch == len(train_dataloader):
                    current_loss = loss.detach().item()
                    average_loss = (epoch_loss / step_in_epoch).item()
                    batch_iterator.set_postfix(
                        loss=f"{current_loss:.3f}",
                        avg=f"{average_loss:.3f}",
                        lengths=f"{encoder_input.size(1)}/{decoder_input.size(1)}",
                    )
                    writer.add_scalar("train/loss", current_loss, global_step)
                    writer.add_scalar("train/average_loss", average_loss, global_step)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - epoch_start
            average_loss = (epoch_loss / len(train_dataloader)).item()
            padding_efficiency = real_tokens / padded_tokens
            writer.add_scalar("train/epoch_loss", average_loss, epoch)
            writer.add_scalar("train/tokens_per_second", real_tokens / epoch_seconds, epoch)
            writer.add_scalar("train/padding_efficiency", padding_efficiency, epoch)
            writer.flush()

            print(
                f"Epoch {epoch:02d} complete | loss {average_loss:.3f} | {_format_duration(epoch_seconds)} | "
                f"{examples_seen / epoch_seconds:.1f} examples/s | {real_tokens / epoch_seconds:,.0f} tokens/s | "
                f"padding efficiency {padding_efficiency:.1%}"
            )

            checkpoint_path = get_weights_file_path(config, f"{epoch:02d}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "global_step": global_step,
                    "config": dict(config),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

            evaluate_validation_loss(
                model,
                validation_loss_dataloader,
                tokenizer_tgt,
                device,
                amp_dtype,
                batch_iterator.write,
                global_step,
                writer,
            )
            run_validation(
                model,
                validation_generation_dataloader,
                tokenizer_tgt,
                min(config["validation_max_len"], config["seq_len"]),
                device,
                amp_dtype,
                batch_iterator.write,
                global_step,
                writer,
                validation_length_buckets,
                config["validation_beam_size"],
                config["validation_length_penalty"],
                config["validation_no_repeat_ngram_size"],
            )

            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured translation Transformer.")
    parser.add_argument("--config", type=str, help="TOML experiment file; defaults to configs/en_it_bpe.toml.")
    parser.add_argument("--epochs", type=int, help="Total epoch count, including any resumed epochs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--no-resume", action="store_true", help="Ignore optimized-run checkpoints and train from scratch.")
    parser.add_argument("--sdpa-attention", action="store_true", help="Use PyTorch SDPA instead of the faster measured MPS path.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the latest checkpoint without training.")
    parser.add_argument("--validation-beam-size", type=int)
    parser.add_argument("--validation-max-length", type=int)
    parser.add_argument("--validation-examples-per-bucket", type=int)
    parser.add_argument("--validation-no-repeat-ngram-size", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["lr"] = args.learning_rate
    if args.sequence_length is not None:
        config["seq_len"] = args.sequence_length
    if args.precision is not None:
        config["precision"] = args.precision
    if args.no_resume:
        config["preload"] = None
    if args.sdpa_attention:
        config["use_fused_attention"] = True
    if args.validation_beam_size is not None:
        config["validation_beam_size"] = args.validation_beam_size
    if args.validation_max_length is not None:
        config["validation_max_len"] = args.validation_max_length
    if args.validation_examples_per_bucket is not None:
        config["validation_examples_per_bucket"] = args.validation_examples_per_bucket
    if args.validation_no_repeat_ngram_size is not None:
        config["validation_no_repeat_ngram_size"] = args.validation_no_repeat_ngram_size

    train_model(config, args.device, evaluate_only=args.evaluate_only)


if __name__ == "__main__":
    main()
