import tomllib
from pathlib import Path
from typing import Literal, TypedDict, cast

from tokenization import TokenizerType


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "en_it_bpe.toml"


class TransformerConfig(TypedDict):
    name: str
    batch_size: int
    num_epochs: int
    lr: float
    seq_len: int
    d_model: int
    datasource: str
    dataset_config: str
    lang_src: str
    lang_tgt: str
    model_basename: str
    preload: str | None
    checkpoint_folder: str
    tokenizer_folder: str
    tokenizer_filename: str
    experiment_folder: str
    tokenizer_type: TokenizerType
    tokenizer_vocab_size: int
    tokenizer_min_frequency: int
    seed: int
    precision: Literal["auto", "float32", "float16", "bfloat16"]
    pad_to_multiple_of: int
    bucket_size_multiplier: int
    log_every: int
    use_fused_attention: bool
    tie_target_embeddings: bool
    validation_max_len: int
    validation_examples_per_bucket: int
    validation_example_indices: list[int]
    validation_beam_size: int
    validation_length_penalty: float
    validation_no_repeat_ngram_size: int


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve CLI paths from the current directory, then from this project directory."""
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    path = Path(config_path).expanduser()
    if path.is_file():
        return path.resolve()
    project_relative_path = PROJECT_DIR / path
    if project_relative_path.is_file():
        return project_relative_path.resolve()
    raise FileNotFoundError(f"Config file does not exist: {config_path}")


def load_config(config_path: str | Path | None = None) -> TransformerConfig:
    """Load one complete experiment configuration from a TOML file."""
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    required_keys = set(TransformerConfig.__required_keys__)
    missing_keys = sorted(required_keys - raw_config.keys())
    unexpected_keys = sorted(raw_config.keys() - required_keys)
    if missing_keys or unexpected_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing {', '.join(missing_keys)}")
        if unexpected_keys:
            problems.append(f"unexpected {', '.join(unexpected_keys)}")
        raise ValueError(f"Invalid config {resolved_path}: {'; '.join(problems)}")

    tokenizer_type = raw_config["tokenizer_type"]
    if tokenizer_type not in {"wordlevel", "byte_bpe"}:
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type!r} in {resolved_path}")
    return cast(TransformerConfig, raw_config)


def _project_path(configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def get_weights_folder_path(config: TransformerConfig) -> Path:
    return _project_path(config["checkpoint_folder"])


def get_weights_file_path(config: TransformerConfig, epoch: int | str) -> Path:
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return get_weights_folder_path(config) / model_filename


def latest_weights_file_path(config: TransformerConfig) -> Path | None:
    """Return the most recently written checkpoint, if one exists."""
    pattern = f"{config['model_basename']}*.pt"
    weights_files = [path for path in get_weights_folder_path(config).glob(pattern) if path.is_file()]
    return max(weights_files, key=lambda path: path.stat().st_mtime, default=None)


def get_tokenizer_file_path(config: TransformerConfig, language: str) -> Path:
    filename = config["tokenizer_filename"].format(language=language)
    return _project_path(config["tokenizer_folder"]) / filename


def get_experiment_path(config: TransformerConfig) -> Path:
    return _project_path(config["experiment_folder"])
