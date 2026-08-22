# Transformer from scratch

An educational encoder-decoder Transformer written in PyTorch and trained to translate English into Italian. The implementation includes word-level tokenization, causal masking, training and validation, checkpointing, a command-line translator, and training/inference notebooks.

## Setup

From the repository root:

```bash
uv sync
```

No manual dataset download is required. The first training run downloads the `opus_books` English-Italian split through Hugging Face Datasets and caches it locally.

## Train

The direct command is:

```bash
uv run python follow/transformer_from_scratch/train.py
```

The Apple-Silicon defaults train for 20 total epochs with a batch size of 64 and BF16 mixed precision. To force a fresh optimized run:

```bash
uv run python follow/transformer_from_scratch/train.py --no-resume
```

Useful options:

- `--epochs N` sets the total target epoch count, including resumed epochs.
- `--batch-size N` lowers or raises memory use.
- `--learning-rate RATE` overrides the default `3e-4`.
- `--precision auto|bfloat16|float16|float32` controls mixed precision; `auto` selects BF16 on MPS.
- `--sequence-length N` changes the maximum supported sequence length.
- `--device auto|mps|cuda|cpu` selects the compute device.
- `--no-resume` starts from random weights instead of loading the latest optimized checkpoint.

Training resumes from the newest checkpoint by default. If the newest file is `tmodel_04.pt` and `--epochs 10` is used, training continues with epoch 05 and finishes after epoch 09.

To evaluate the latest checkpoint without training another epoch:

```bash
uv run python follow/transformer_from_scratch/train.py --evaluate-only
```

Evaluation now reports token-weighted, teacher-forced loss over the entire validation split. It then generates a fixed set drawn from the short, medium, and long thirds of that split, so qualitative changes are compared against representative examples rather than whichever two rows happen to come first. The default generation limit is the full 320 tokens.

For notebook-based training, launch Jupyter from this directory and open `training.ipynb`:

```bash
cd follow/transformer_from_scratch
uv run jupyter lab
```

The notebook exposes the same configuration, training loop, and TensorBoard logs as the CLI.

## Training outputs

All generated files are anchored to this directory, regardless of the shell's current working directory:

- `tokenizer_en.json` and `tokenizer_it.json` contain the learned vocabularies.
- `opus_books_weights_mps/tmodel_XX.pt` contains one optimized checkpoint per epoch.
- `runs/tmodel_mps/` contains TensorBoard events.

These artifacts are required for inference. The model architecture settings in `config.py`, especially `seq_len` and `d_model`, must match the checkpoint used for inference.

## Run inference

After at least one epoch has produced a checkpoint:

```bash
uv run python follow/transformer_from_scratch/translate.py "I am a student."
```

This uses a beam width of 3, length normalization, and 3-gram repetition blocking. Useful controls include:

- `--beam-size 1` switches to greedy decoding; values from 3 to 5 are normally enough.
- `--no-repeat-ngram-size 0` disables repetition blocking.
- `--repetition-penalty 1.1` additionally lowers the score of tokens already used.
- `--length-penalty 0.6` controls beam search's preference for longer candidates.
- `--max-length 320` controls the output limit.
- `--checkpoint PATH` selects a checkpoint and `--device cpu` forces CPU inference.

For interactive use, open `inference.ipynb`, load the model once, and edit the `sentence` variable.

Generation starts with `[SOS]` and predicts until `[EOS]` or the length limit. It uses an incremental KV cache: each decoder layer stores previously projected self-attention keys/values and stores its encoder cross-attention keys/values once. The decoder therefore processes only the new token instead of recomputing the entire prefix. Beam search keeps several promising partial translations and reorders their caches whenever a different parent beam wins.

The current `WordLevel` tokenizer splits punctuation into separate tokens. A small punctuation-aware cleanup step removes display artifacts such as `word ,` and `Arkad ’ ic`; this changes formatting, not the model's underlying token choices.

## Files

- `model.py` implements embeddings, positional encoding, attention, encoder/decoder blocks, and the final vocabulary projection.
- `dataset.py` tokenizes sentence pairs and builds encoder padding masks plus decoder padding/causal masks.
- `config.py` contains typed settings and artifact paths.
- `train.py` downloads data, builds tokenizers, trains, validates, logs metrics, and saves/resumes checkpoints.
- `translate.py` contains cached greedy/beam decoding, repetition controls, punctuation cleanup, checkpoint loading, and the translation CLI.
- `training.ipynb` runs and monitors training interactively.
- `inference.ipynb` loads a trained checkpoint and translates custom sentences.

## Tensor shapes

With batch size `B`, sequence lengths `S`/`T`, model width `D`, head count `H`, and vocabulary size `V`:

| Value | Shape |
| --- | --- |
| Source/target token IDs | `(B, S)` / `(B, T)` |
| Embedded residual stream | `(B, S, D)` / `(B, T, D)` |
| Attention queries/keys/values | `(B, H, length, D/H)` |
| Attention scores | `(B, H, query_length, key_length)` |
| Decoder output | `(B, T, D)` |
| Vocabulary logits | `(B, T, V)` |

The projection layer returns raw logits because `CrossEntropyLoss` applies the required log-softmax internally.

## Apple Silicon optimizations

The training path is tuned for this project's M5 Max machine:

- BF16 automatic mixed precision instead of FP32-only matrix multiplication.
- Batch size 64 to better occupy the 40-core GPU.
- One-time in-memory tokenization instead of re-tokenizing every example every epoch.
- Dynamic padding rounded to multiples of 8 instead of padding every sentence to 350 tokens.
- Length-bucketed batches, giving about 77% measured padding efficiency versus roughly 8% previously.
- Native fused LayerNorm and tied target embedding/projection weights.
- Explicit matmul attention, which benchmarked about 19% faster than PyTorch SDPA on this MPS setup.
- Fused Adam and fewer GPU synchronizations from progress/TensorBoard logging.
- MPS fast-math and native Metal matmul preferences enabled before PyTorch initializes.
- KV-cached incremental decoding for validation and inference (training remains fully parallel and does not need a cache).

On the local 40-core M5 Max, a 100-batch end-to-end benchmark measured about 377 examples/second and estimated roughly 1.3 minutes of training compute per epoch. Checkpoint saving and validation add some time.
