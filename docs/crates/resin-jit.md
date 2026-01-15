# resin-jit

Generic JIT compilation infrastructure using Cranelift.

## What It Does

Compiles graph nodes and expressions to native machine code at runtime. Provides both scalar and SIMD compilation modes.

## Key Types

- `JitCompiler` - Cranelift-based compiler instance
- `JitCompilable` - Trait for nodes that can emit IR
- `SimdCompilable` - Extension for SIMD-capable nodes
- `JitCategory` - Classification: `PureMath`, `Stateful`, `External`
- `CompiledScalar` - Result type for `fn(f32) -> f32`
- `CompiledSimdBlock` - Result type for block processing

## Performance

For 44100 samples (1 second of audio):

| Mode | Time | Notes |
|------|------|-------|
| SIMD JIT | 5.1 µs | 6.6x faster than native Rust |
| Native Rust | 31.4 µs | Bounds checking overhead |
| Scalar JIT | 209 µs | No vectorization |

SIMD JIT beats native because:
1. Zero bounds checking (raw pointers)
2. Explicit f32x4 vectorization
3. 4x fewer loop iterations

## Usage

```rust
use rhizome_resin_jit::{JitCompiler, JitConfig};

// Compile affine transform: y = 0.5x + 1.0
let mut compiler = JitCompiler::new(JitConfig::default())?;

// Scalar: process one sample
let scalar = compiler.compile_affine(0.5, 1.0)?;
let y = scalar.call_f32(2.0);  // 2.0

// SIMD: process buffer
let simd = compiler.compile_affine_simd(0.5, 1.0)?;
let input = vec![0.0, 1.0, 2.0, 3.0];
let mut output = vec![0.0; 4];
simd.process(&input, &mut output);  // [1.0, 1.5, 2.0, 2.5]
```

## Implementing JitCompilable

Domain crates implement the traits for their node types:

```rust
impl JitCompilable for MyNode {
    fn jit_category(&self) -> JitCategory {
        JitCategory::PureMath  // or Stateful, External
    }

    fn emit_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder,
        ctx: &mut JitContext,
    ) -> Vec<Value> {
        // Emit Cranelift IR
        let result = builder.ins().fmul(inputs[0], self.factor);
        vec![result]
    }
}
```

## Node Categories

| Category | Behavior | SIMD |
|----------|----------|------|
| `PureMath` | Fully inlined | Yes |
| `Stateful` | Callback to Rust | No (breaks batching) |
| `External` | Call external fn | Depends |

## Related Crates

- `resin-audio` - Implements `JitCompilable` for audio nodes
- `resin-core` - Graph types that can be compiled

## Feature Flags

- `cranelift` - Enables JIT compilation (required)

Without the feature, stub types are provided for API compatibility.
