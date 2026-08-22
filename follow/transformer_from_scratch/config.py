from pathlib import Path
from typing import Literal, TypedDict


PROJECT_DIR = Path(__file__).resolve().parent


class TransformerConfig(TypedDict):
    batch_size: int
    num_epochs: int
    lr: float
    seq_len: int
    d_model: int
    datasource: str
    lang_src: str
    lang_tgt: str
    model_folder: str
    model_basename: str
    preload: str | None
    tokenizer_file: str
    experiment_name: str
    seed: int
    precision: Literal["auto", "float32", "float16", "bfloat16"]
    pad_to_multiple_of: int
    bucket_size_multiplier: int
    log_every: int
    use_fused_attention: bool
    tie_target_embeddings: bool
    validation_max_len: int
    validation_examples_per_bucket: int
    validation_beam_size: int
    validation_length_penalty: float
    validation_no_repeat_ngram_size: int


def get_config() -> TransformerConfig:
    """Return the default English-to-Italian training configuration."""
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 3e-4,
        "seq_len": 320,
        "d_model": 512,
        "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights_mps",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_mps",
        "seed": 1337,
        "precision": "auto",
        "pad_to_multiple_of": 8,
        "bucket_size_multiplier": 10,
        "log_every": 20,
        # Explicit matmul attention benchmarks faster than SDPA on MPS 2.13.
        "use_fused_attention": False,
        "tie_target_embeddings": True,
        "validation_max_len": 320,
        "validation_examples_per_bucket": 2,
        "validation_beam_size": 3,
        "validation_length_penalty": 0.6,
        "validation_no_repeat_ngram_size": 3,
    }


def get_weights_folder_path(config: TransformerConfig) -> Path:
    dataset_name = config["datasource"].rsplit("/", maxsplit=1)[-1]
    return PROJECT_DIR / f"{dataset_name}_{config['model_folder']}"


def get_weights_file_path(config: TransformerConfig, epoch: int | str) -> Path:
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return get_weights_folder_path(config) / model_filename


def latest_weights_file_path(config: TransformerConfig) -> Path | None:
    """Return the most recently written checkpoint, if one exists."""
    pattern = f"{config['model_basename']}*.pt"
    weights_files = [path for path in get_weights_folder_path(config).glob(pattern) if path.is_file()]
    return max(weights_files, key=lambda path: path.stat().st_mtime, default=None)


def get_tokenizer_file_path(config: TransformerConfig, language: str) -> Path:
    return PROJECT_DIR / config["tokenizer_file"].format(language)


def get_experiment_path(config: TransformerConfig) -> Path:
    return PROJECT_DIR / config["experiment_name"]
