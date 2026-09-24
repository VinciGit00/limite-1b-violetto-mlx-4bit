# Violetto on the Mac mini: Ollama compatibility

**Ollama import is blocked, not completed.** On the Mac mini M4, Ollama 0.34.2
rejects the converted model with:

```text
Error: unsupported MLX architecture: model "LimiteForCausalLM"
```

The model files are stored at:

```text
/Volumes/MV work/ollama/imports/limite-1b-violetto-mlx-4bit
```

This is inside the existing `~/.ollama` storage location. Storing the files there
does not register a working Ollama model. We do not force-create a broken
manifest or publish an unusable Ollama registry entry.

Use the installed MLX launcher on the Mac mini instead:

```bash
~/.ollama/imports/limite-1b-violetto-mlx-4bit/run_mlx.sh \
  --prompt "What is 17 multiplied by 6?"
```

The launcher uses the isolated environment prepared for the executed notebook,
with the pinned Limite adapter. It uses the saved original chat template,
temperature 0.6, top-p 0.95, and a default 3,000-token budget. It does not route
through the Ollama server. The current Ollama installation and other models
remain available.

`Modelfile` records the failed import recipe for future compatibility testing;
it is not a claim of working support. The upstream Ollama MLX architecture
registry at v0.34.2 does not contain Limite:
https://github.com/ollama/ollama/blob/v0.34.2/mlxrunner/model/architectures/architectures.go

Native Ollama use requires implementing and validating this architecture in
Ollama's runtime. A GGUF container alone does not provide that implementation.
