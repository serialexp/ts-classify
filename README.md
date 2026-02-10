# ts-classify

Fast text classification via SVM, compiled to WebAssembly.

## Features

- **Multiclass classification** via One-vs-Rest (OvR) or One-vs-One (OvO)
- **Binary SVM** with coordinate descent solver (O(N) per pass)
- **Sparse vectors** - efficient for high-dimensional data (text, trigrams)
- **Binary serialization** - train once, load instantly
- **Written in Rust**, compiled to WASM for near-native performance in the browser

## Installation

```bash
npm install ts-classify
```

## Usage

```javascript
import init, { JsOneVsRestSVM } from 'ts-classify';

await init();

const svm = new JsOneVsRestSVM();

// Flat sparse format: [idx, val, idx, val, ...]
const samplesFlat = new Float64Array([
    0, -2.0, 1, 0.0,  // sample 0: 2 pairs
    0, 1.0, 1, 1.0,   // sample 1: 2 pairs
]);
const sampleLengths = new Uint32Array([2, 2]);  // pairs per sample
const labels = new Int32Array([0, 1]);

svm.train(samplesFlat, sampleLengths, labels);

const prediction = svm.predict(new Float64Array([0, -2.0, 1, 0.0]));

// Serialize/deserialize
const bytes = svm.to_bytes();
const loaded = JsOneVsRestSVM.from_bytes(bytes);
```

## Performance

Scaling (coordinate descent, OvR):

| Classes | Classifiers | Train | Predict |
|---------|-------------|-------|---------|
| 10 | 10 | 511µs | 144ns |
| 50 | 50 | 560µs | 743ns |
| 100 | 100 | 1.4ms | 1.6µs |
| 5000 | 5000 | ~50-100ms | ~80µs |

## Building from source

Requires [Rust](https://rustup.rs/) and [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/).

```bash
wasm-pack build --target web --no-default-features
```

Run tests (native):

```bash
cargo test --release
```

## License

MIT
