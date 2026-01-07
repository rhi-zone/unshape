# Getting Started

## Prerequisites

- Rust (edition 2024)
- Nix with flakes (optional, for dev environment)

## With Nix

```bash
# Enter dev shell (auto-detected via direnv, or manually)
nix develop

# Build
cargo build
```

## Without Nix

Ensure you have:
- `rustc` and `cargo` (rustup recommended)
- `mold` or `lld` for fast linking (optional)

```bash
cargo build
```

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
resin = "0.1"

# Or individual crates
resin-mesh = "0.1"
resin-audio = "0.1"
```

## Documentation

```bash
cd docs
bun install
bun run dev
```

Opens at `http://localhost:5173/resin/`
