# ts-classify

Fast text classification via SVM, compiled to WebAssembly.

## Features

- **Sparse vectors** - efficient for high-dimensional data (text, trigrams)
- **Binary serialization** - train once, load instantly
- **Written in Rust**, compiled to WASM for near-native performance in the browser

## Installation

```bash
npm install ts-classify
```

## Classifiers

All classifiers use the same sparse data format: flat `Float64Array` of `[index, value, index, value, ...]` pairs.

```javascript
import init, { JsOneVsRestSVM, JsMulticlassSVM, JsNearestCentroid, JsSVM } from 'ts-classify';

await init();
```

### JsOneVsRestSVM

Multiclass SVM using one-vs-rest strategy. Trains K classifiers for K classes. Best for large numbers of classes.

```javascript
const svm = new JsOneVsRestSVM();
svm.train(samplesFlat, sampleLengths, labels);
const prediction = svm.predict(sampleFlat);
const margins = svm.margins(sampleFlat); // [class0, margin0, class1, margin1, ...]
```

### JsMulticlassSVM

Multiclass SVM using one-vs-one strategy. Trains K(K-1)/2 classifiers, predicts by voting. Often more accurate for smaller numbers of classes.

```javascript
const svm = new JsMulticlassSVM();
svm.train(samplesFlat, sampleLengths, labels);
const prediction = svm.predict(sampleFlat);
```

### JsNearestCentroid

Computes the centroid of each class and predicts by cosine similarity. Very fast to train, no hyperparameters.

```javascript
const nc = new JsNearestCentroid();
nc.train(samplesFlat, sampleLengths, labels);
const prediction = nc.predict(sampleFlat);
const similarities = nc.margins(sampleFlat); // [class0, sim0, class1, sim1, ...]
```

### JsSVM

Binary SVM for two-class problems. Labels are -1.0 or 1.0.

```javascript
const svm = new JsSVM();
svm.train(samplesFlat, sampleLengths, labels); // labels: Float64Array of -1.0/1.0
const prediction = svm.predict(sampleFlat);     // returns -1.0 or 1.0
const margin = svm.margin(sampleFlat);           // raw decision value
```

### Data format

```javascript
// Flat sparse format: [idx, val, idx, val, ...]
const samplesFlat = new Float64Array([
    0, -2.0, 1, 0.0,  // sample 0: 2 pairs
    0, 1.0, 1, 1.0,   // sample 1: 2 pairs
]);
const sampleLengths = new Uint32Array([2, 2]);  // pairs per sample
const labels = new Int32Array([0, 1]);           // class IDs (or Float64Array for JsSVM)
```

### Serialization

All classifiers support binary serialization:

```javascript
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
