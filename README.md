# Limite 1B Violetto — MLX 4-bit on Mac mini M4

[Notebook](01_violetto_mlx_quantization.ipynb) · [Model on Hugging Face](https://huggingface.co/vinci00/limite-1b-violetto-mlx-4bit)

Convert [Paradigma's original Violetto checkpoint](https://huggingface.co/paradigma-inc/limite-1b-violetto)
to an unquantized MLX baseline and an affine 4-bit derivative, then evaluate both
on the same mathematical prompts and held-out GSM8K problems. This conversion adds no training or fine-tuning. The release is published as
[`vinci00/limite-1b-violetto-mlx-4bit`](https://huggingface.co/vinci00/limite-1b-violetto-mlx-4bit),
with reproducible code in
[`VinciGit00/limite-1b-violetto-mlx-4bit`](https://github.com/VinciGit00/limite-1b-violetto-mlx-4bit).

| Property | Value |
| --- | --- |
| Source | `paradigma-inc/limite-1b-violetto` |
| Pinned source revision | `9402e422e87b220507963cd42c997459c1e87b43` |
| Architecture | Custom `limite`, 48 layers, about 1.035B parameters |
| Task | Text-only, single-turn mathematical reasoning |
| Source precision | BF16 matrices plus FP32 auxiliary tensors |
| Quantization | MLX affine 4-bit, group size 64, including tied embedding/head |
| License | Apache-2.0, model and community adapter |
| Reference hardware | Mac mini M4, 16 GiB unified memory |
| Runtime | Apple Silicon, `mlx==0.32.2`, `mlx-lm==0.31.3` |

Violetto requires a custom architecture adapter: stock mlx-lm 0.31.3 does not
support `limite`. The requirements pin
[`pierjoe/limite-mlx`](https://huggingface.co/pierjoe/limite-mlx/tree/b21b7a4fa6ca05cb02082d9f7819b8d1387f2f1a)
version 0.1.0 to repository revision
`b21b7a4fa6ca05cb02082d9f7819b8d1387f2f1a`. This third-party package registers
only `mlx_lm.models.limite` through a Python startup import hook. It does not
modify mlx-lm's installed files. Both variants use this same implementation.
The official runtime is [Paradigma's vLLM plugin](https://github.com/paradigma-inc/limite-violetto).
We have not independently reproduced the adapter author's numerical-equivalence
claims against that runtime.

## Run the notebook

The reference execution runs on the Mac mini M4, not the development MacBook.
Each execution records hardware and software in the locally generated `benchmark_artifacts/hardware.json`.
After cloning this repository on the Mac mini:

```bash
cd limite-1b-violetto-mlx-4bit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 01_violetto_mlx_quantization.ipynb
```

Allow about 5 GiB of disk space for the source and two converted checkpoints.
Run the notebook from this repository root. Restart an existing notebook kernel after installing the adapter.
Model weights and all generated benchmark reports, charts, provenance, and dataset caches are ignored by Git. The `benchmark_artifacts/` directory is created locally when you run the notebook; it is not included in this repository.

The notebook downloads the pinned source snapshot, converts both variants from
it, reloads the 4-bit output, generates an answer, and runs the paired benchmark.
Completed conversions carry a provenance manifest; an existing directory without
a matching manifest is rejected. Choose a fresh path for an interrupted or
different conversion. Source weights, tokenizer, and template are not replaced
with another author's quantized checkpoint.

The adapter folds the source attention scales into projections and retains the
small FP32 auxiliary tensors. Both converted configurations use end-token IDs
`[151643, 151645]` because the upstream tokenizer and generation configuration
disagree. The original fixed chat template is preserved identically in both
variants.

## Generate from the saved model

```python
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load('./artifacts/limite-1b-violetto-4bit')
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'How many positive divisors does 360 have?'}],
    tokenize=False,
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=3000,
               sampler=make_sampler(temp=0.6, top_p=0.95)))
```

The template supplies Violetto's fixed mathematical system prompt, including
the boxed-answer instruction. Do not substitute a generic system message.
Temperature 0.6 and top-p 0.95 follow the upstream recommendation for interactive
use; the comparison below intentionally uses greedy decoding for both variants.

## Evaluation protocol

The notebook runs a **10-example teaching evaluation**, with seed 42 and 1,024
output tokens per problem. It is not a reproduction of the upstream AIME or
other competition results. Violetto can spend thousands of tokens reasoning;
the output cap can therefore lower measured accuracy.

The local `benchmark_mlx.py` and `benchmark_accuracy.py` evaluate Violetto's
mathematical prompts and fixed chat template. Scoring uses the last complete
boxed field after `</think>`, allows LaTeX delimiters and trailing prose outside
the box, and requires the box itself to contain only a number. An unfinished
thinking section scores zero. Missing, unfinished, nonnumeric, and malformed final
answers count as incorrect; numbers in reasoning do not count as final answers.

Both variants receive the same unmodified questions, chat template, greedy
decoding, token cap, and sorted random sample of GSM8K's test split. Dataset
revision `3101c7d5072418e28b9008a6636bde82a006892c` and the downloaded JSONL's
SHA-256 are checked. No training is performed; the training split is unused and
the test split is not used for selecting prompts or quantization settings.

Reproduce the teaching evaluation:

```bash
python benchmark_mlx.py \
  --baseline-model artifacts/limite-1b-violetto-bf16-mlx \
  --model artifacts/limite-1b-violetto-4bit \
  --max-tokens 1024 --accuracy-samples 10 --accuracy-max-tokens 1024 \
  --seed 42 --output-dir benchmark_artifacts
```

For a separately planned larger evaluation, choose `--accuracy-samples 100`
and `--accuracy-max-tokens 3000` before examining test outcomes. This can take
considerably longer. Four fixed mathematical smoke questions measure runtime
and basic functionality separately from the held-out accuracy sample. A failed
smoke case yields exit code 1 after reports are saved; it is not evidence that
conversion failed.

Outputs include numerical runtime and MLX peak memory, correct/total, accuracy,
95% Wilson intervals, paired changes and bootstrap uncertainty, per-example
predictions, and charts. Conversion provenance records source/adapter revisions
and converted weight checksums. Runtime excludes loading and tokenization;
generated lengths can differ. A small-sample tie does not demonstrate general
quality parity. Pretraining contamination is unknown.

## Scope and attribution

Use this workflow for local mathematical reasoning experiments. The upstream
model is lightly instruction-tuned and is not presented as a general assistant.
The comparison measures quantization inside the community MLX runtime, not its
equivalence to official vLLM. No image inference or GGUF quality measurement is included. The attempted
Ollama 0.34.2 import failed because `LimiteForCausalLM` is unsupported; see the
[Ollama compatibility record and installed MLX launcher](ollama/README.md). MLX measurements must not be attributed
to another runtime or artifact.

Credit for Violetto belongs to Paradigma; the MLX architecture port is by
`pierjoe`. Retain the Apache-2.0 license and upstream attribution when
redistributing. See the [upstream model card](https://huggingface.co/paradigma-inc/limite-1b-violetto)
and [adapter source](https://huggingface.co/pierjoe/limite-mlx/blob/b21b7a4fa6ca05cb02082d9f7819b8d1387f2f1a/limite.py).

## Measured standalone run — Mac mini M4, 2026-09-24

All notebook code cells completed on Mac mini M4 with 16 GiB unified memory.
The paired GSM8K test uses 10 examples, seed 42, greedy decoding, and a maximum
of 1,024 generated tokens per example.

| Variant | GSM8K correct/total | Mathematical smoke checks | Mean smoke latency | Output tokens/s | MLX peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLX unquantized | 5/10 | 100% | 12.16 s | 40.22 | 1.974 GiB |
| MLX 4-bit | 5/10 | 100% | 6.12 s | 88.62 | 0.599 GiB |

Both variants have four invalid final answers and one additional incorrect
answer. The 95% Wilson accuracy interval is 23.66%–76.34%. All paired binary
outcomes agree, so the delta and paired bootstrap interval are zero; this
small, token-limited sample does not establish quality parity.

Weights occupy 1.929 GiB for the source-precision baseline and 0.543 GiB for
4-bit. Latency excludes loading; generated lengths differ. These measurements
come from this standalone rerun and can differ from earlier runs on the same Mac.
