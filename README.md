# SVM

A fast Support Vector Machine implementation in Rust, targeting both native and WebAssembly.

## Features

- **Binary SVM** with two solvers:
  - **Coordinate Descent** (default) - O(N) per pass, fast for large datasets
  - **SMO** - O(N²), useful for kernel SVMs (future)
- **Multiclass classification**:
  - **One-vs-Rest (OvR)** - K classifiers for K classes, fast prediction
  - **One-vs-One (OvO)** - K(K-1)/2 classifiers, often more accurate
- **Sparse vectors** - efficient for high-dimensional data (text, trigrams)
- **Parallel training** via rayon (native only)
- **Binary serialization** via bincode
- **WebAssembly support** with JS-friendly API

## Installation

```toml
[dependencies]
svm = { path = "." }
```

## Usage

### Binary SVM

```rust
use svm::{SVM, SparseVec};

// Sparse format: Vec<(index, value)>
let samples: Vec<SparseVec> = vec![
    vec![(0, -2.0), (1, 0.0)],
    vec![(0, -1.0), (1, 1.0)],
    vec![(0, 1.0), (1, 0.0)],
    vec![(0, 2.0), (1, 1.0)],
];
let labels = vec![-1.0, -1.0, 1.0, 1.0];

let mut svm = SVM::new();
svm.train_sparse(&samples, &labels);

assert_eq!(svm.predict_sparse(&[(0, -3.0)]), -1.0);
assert_eq!(svm.predict_sparse(&[(0, 3.0)]), 1.0);
```

### Multiclass (One-vs-Rest)

```rust
use svm::{OneVsRestSVM, SparseVec};

let samples: Vec<SparseVec> = vec![
    vec![(0, -2.0), (1, -2.0)],  // class 0
    vec![(0, 2.0), (1, -2.0)],   // class 1
    vec![(0, 0.0), (1, 2.0)],    // class 2
    // ... more samples
];
let labels = vec![0, 1, 2, /* ... */];

let mut svm = OneVsRestSVM::new();
svm.train_sparse(&samples, &labels);

let predicted_class = svm.predict_sparse(&[(0, 0.0), (1, 3.0)]);
```

### Custom Configuration

```rust
use svm::{SVM, TrainConfig, Solver};

let config = TrainConfig::coordinate_descent()
    .with_c(0.5)           // regularization (default: 1.0)
    .with_max_iter(500)    // max iterations (default: 1000)
    .with_tol(0.001);      // tolerance (default: 0.01)

let mut svm = SVM::new();
svm.train_sparse_with_config(&samples, &labels, &config);

// Or use SMO solver
let config = TrainConfig::smo()
    .with_c(1.0)
    .with_max_iter(10000);
```

### Serialization

```rust
// Save
let bytes = svm.to_bytes()?;
std::fs::write("model.bin", &bytes)?;

// Load
let bytes = std::fs::read("model.bin")?;
let svm = SVM::from_bytes(&bytes)?;
```

## WebAssembly

Build for WASM (without parallel feature):

```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features
```

Or use wasm-pack:

```bash
wasm-pack build --target web --no-default-features
```

### JavaScript API

```javascript
import { JsOneVsRestSVM } from 'svm';

const svm = new JsOneVsRestSVM();

// Flat sparse format: [idx, val, idx, val, ...]
const sampleFlat = new Float64Array([0, -2.0, 1, 0.0]);  // single sample
const samplesFlat = new Float64Array([
    0, -2.0, 1, 0.0,  // sample 0: 2 pairs
    0, 1.0, 1, 1.0,   // sample 1: 2 pairs
]);
const sampleLengths = new Uint32Array([2, 2]);  // pairs per sample
const labels = new Int32Array([0, 1]);

svm.train(samplesFlat, sampleLengths, labels);

const prediction = svm.predict(sampleFlat);

// Serialize/deserialize
const bytes = svm.to_bytes();
const loaded = JsOneVsRestSVM.from_bytes(bytes);
```

## Performance

Coordinate descent vs SMO on 50-class OvR:

| Solver | Train Time |
|--------|------------|
| Coordinate Descent | 6ms |
| SMO | 29s |

Scaling (coordinate descent, OvR):

| Classes | Classifiers | Train | Predict |
|---------|-------------|-------|---------|
| 10 | 10 | 511µs | 144ns |
| 50 | 50 | 560µs | 743ns |
| 100 | 100 | 1.4ms | 1.6µs |
| 5000 | 5000 | ~50-100ms | ~80µs |

## Feature Flags

- `parallel` (default) - Enable parallel training via rayon. Disable for WASM.

```toml
# Native with parallel
svm = { path = "." }

# WASM (no parallel)
svm = { path = ".", default-features = false }
```

## License

MIT
