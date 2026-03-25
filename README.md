# preparate

A command-line tool for inspecting and rearranging transformer layers in [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) model files.

## What it does

**preparate** lets you:

- **Inspect** a GGUF file's metadata, tensor layout, and per-layer sizes
- **Rearrange layers** — duplicate, reorder, or remove transformer blocks to produce a new GGUF file with renumbered layers

This is useful for experimenting with model architecture — for example, duplicating certain transformer blocks to see how it affects model behavior, or pruning layers to create smaller variants.

## Building

Requires Rust (edition 2024).

```sh
cargo build --release
```

## Usage

### Inspect a model

```sh
preparate info model.gguf
```

Displays:
- GGUF version and alignment
- All metadata key/value pairs
- Non-block tensors (embeddings, output weights, norms)
- Block tensors grouped by layer, with quantization types, dimensions, and sizes

### Rearrange layers

```sh
preparate merge <source.gguf> <output.gguf> <layer-list>
```

The layer list is a comma-separated sequence of source layer numbers. Ranges are supported. Layers are renumbered sequentially (0, 1, 2, ...) in the output.

**Examples:**

```sh
# Duplicate layer 3
preparate merge model.gguf out.gguf 0,1,2,3,3,4,5

# Use a range
preparate merge model.gguf out.gguf 0-3,3,4-7

# Drop layers 2 and 5 from an 8-layer model
preparate merge model.gguf out.gguf 0,1,3,4,6,7
```

Non-block tensors (token embeddings, output head, etc.) are copied unchanged. The `block_count` metadata field is updated automatically.

## How it works

The tool memory-maps the source GGUF file, parses the header/metadata/tensor info, then writes a new GGUF with the selected block tensors in the specified order. Tensor data is copied verbatim — no re-quantization or conversion is performed.

## License

See repository for license information.
