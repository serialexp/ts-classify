# Current Task: SVM Rust Implementation

## Context

Bart wanted to build an SVM library in Rust for classifying manufacturer names into ~5000 canonical categories. The original svmjs (Karpathy's JS library) was binary-only and wouldn't scale.

## What We Built

A complete SVM implementation in Rust with:

1. **Binary SVM** with two solvers:
   - SMO (Sequential Minimal Optimization) - O(N²), for kernels
   - Coordinate Descent - O(N) per pass, 500x faster for linear SVM

2. **Multiclass**:
   - One-vs-One (OvO) - K(K-1)/2 classifiers
   - One-vs-Rest (OvR) - K classifiers, better for large K

3. **Features**:
   - Sparse vector support (`Vec<(usize, f64)>`)
   - Parallel training via rayon (optional feature)
   - Configurable: solver, C, max_iter, tolerance
   - Binary serialization via bincode/serde

4. **WASM Support**:
   - JS-friendly wrapper types (`JsSVM`, `JsOneVsRestSVM`)
   - Flat array API for sparse vectors
   - Builds for `wasm32-unknown-unknown`

## Current State

- All 21 tests pass
- Compiles for both native (with rayon) and WASM (without rayon)
- WASM build succeeds: `cargo build --release --target wasm32-unknown-unknown --no-default-features`

## What's Left for WASM

1. **Install wasm-pack** and build a proper npm package:
   ```bash
   cargo install wasm-pack
   wasm-pack build --target web --no-default-features
   ```

2. **Test the JS API** - the `JsSVM` and `JsOneVsRestSVM` wrappers haven't been tested from actual JS yet

3. **Consider adding**:
   - TypeScript type definitions
   - Example HTML/JS demo
   - npm publish setup

## Files

- `src/lib.rs` - All implementation code (~1300 lines)
- `Cargo.toml` - Dependencies and feature flags
- `README.md` - Documentation

## Key Design Decisions

- **Coordinate Descent as default** - 500x faster than SMO for linear SVM
- **OvR for 5000 classes** - 5000 classifiers vs 12.5M for OvO, ~80µs prediction
- **Sparse vectors** - Efficient for high-dimensional text features (trigrams)
- **Flat arrays for WASM** - `[idx, val, idx, val, ...]` format for JS interop

## Running Tests

```bash
# With parallel (native)
cargo test --release

# Without parallel (WASM mode)
cargo test --release --no-default-features
```
