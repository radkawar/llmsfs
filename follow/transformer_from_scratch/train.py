import argparse
import shutil
from collections.abc import Callable, Iterator, Mapping
from typing import cast

import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm

from config import (
    TransformerConfig,
    get_config,
    get_experiment_path,
    get_tokenizer_file_path,
    get_weights_file_path,
    get_weights_folder_path,
    latest_weights_file_path,
)
from dataset import BilingualBatch, BilingualDataset, BilingualSample, TranslationRows, special_token_id
from model import Transformer, build_transformer
from translate import greedy_decode, select_device


def run_validation(
    model: Transformer,
    validation_ds: DataLoader[BilingualSample],
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: torch.device,
    print_msg: Callable[[str], None],
    global_step: int,
    writer: SummaryWriter | None,
    num_examples: int = 2,
) -> None:
    model.eval()
    expected: list[str] = []
    predicted: list[str] = []
    console_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    with torch.inference_mode():
        for count, raw_batch in enumerate(validation_ds, start=1):
            batch = cast(BilingualBatch, raw_batch)
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            if encoder_input.size(0) != 1:
                raise ValueError("Validation requires a batch size of 1")

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().tolist(), skip_special_tokens=True)

            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count >= num_examples:
                print_msg("-" * console_width)
                break

    if writer is not None and predicted:
        writer.add_scalar("validation/cer", float(CharErrorRate()(predicted, expected)), global_step)
        writer.add_scalar("validation/wer", float(WordErrorRate()(predicted, expected)), global_step)
        writer.add_scalar("validation/bleu", float(BLEUScore()(predicted, expected)), global_step)
        writer.flush()


def get_all_sentences(ds: HFDataset, language: str) -> Iterator[str]:
    for raw_item in ds:
        item = cast(Mapping[str, object], raw_item)
        translations = cast(Mapping[str, str], item["translation"])
        yield translations[language]


def get_or_build_tokenizer(config: TransformerConfig, ds: HFDataset, language: str) -> Tokenizer:
    tokenizer_path = get_tokenizer_file_path(config, language)
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, language), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def get_ds(
    config: TransformerConfig,
) -> tuple[DataLoader[BilingualSample], DataLoader[BilingualSample], Tokenizer, Tokenizer]:
    raw_dataset = load_dataset(
        config["datasource"],
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )
    if not isinstance(raw_dataset, HFDataset):
        raise TypeError("Expected a regular Hugging Face Dataset")

    tokenizer_src = get_or_build_tokenizer(config, raw_dataset, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, raw_dataset, config["lang_tgt"])

    split = raw_dataset.train_test_split(test_size=0.1, seed=config["seed"])
    train_ds_raw = split["train"]
    val_ds_raw = split["test"]
    train_ds = BilingualDataset(
        cast(TranslationRows, train_ds_raw),
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        cast(TranslationRows, val_ds_raw),
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0
    for raw_item in raw_dataset:
        item = cast(Mapping[str, object], raw_item)
        translations = cast(Mapping[str, str], item["translation"])
        max_len_src = max(max_len_src, len(tokenizer_src.encode(translations[config["lang_src"]]).ids))
        max_len_tgt = max(max_len_tgt, len(tokenizer_tgt.encode(translations[config["lang_tgt"]]).ids))

    print(f"Maximum source length: {max_len_src} tokens")
    print(f"Maximum target length: {max_len_tgt} tokens")
    print(f"Training examples: {len(train_ds):,}; validation examples: {len(val_ds):,}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: TransformerConfig, vocab_src_len: int, vocab_tgt_len: int) -> Transformer:
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )


def train_model(config: TransformerConfig, requested_device: str | torch.device | None = None) -> None:
    device = select_device(requested_device)
    print(f"Using {device} device")
    get_weights_folder_path(config).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(str(get_experiment_path(config)))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

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
        initial_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
    else:
        print("No checkpoint found; starting from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=special_token_id(tokenizer_tgt, "[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    try:
        for epoch in range(initial_epoch, config["num_epochs"]):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

            for raw_batch in batch_iterator:
                batch = cast(BilingualBatch, raw_batch)
                encoder_input = batch["encoder_input"].to(device)
                decoder_input = batch["decoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)
                decoder_mask = batch["decoder_mask"].to(device)
                label = batch["label"].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                logits = model.project(decoder_output)
                loss = loss_fn(logits.reshape(-1, tokenizer_tgt.get_vocab_size()), label.reshape(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                global_step += 1

            run_validation(
                model,
                val_dataloader,
                tokenizer_tgt,
                config["seq_len"],
                device,
                batch_iterator.write,
                global_step,
                writer,
            )

            checkpoint_path = get_weights_file_path(config, f"{epoch:02d}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the English-to-Italian Transformer.")
    parser.add_argument("--epochs", type=int, help="Total epoch count, including any resumed epochs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and train from scratch.")
    args = parser.parse_args()

    config = get_config()
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["lr"] = args.learning_rate
    if args.no_resume:
        config["preload"] = None

    train_model(config, args.device)


if __name__ == "__main__":
    main()
