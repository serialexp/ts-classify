// ABOUTME: JS tests for WASM classifier bindings.
// ABOUTME: Tests all four classifiers via the Node.js WASM build.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  JsSVM,
  JsOneVsRestSVM,
  JsMulticlassSVM,
  JsNearestCentroid,
} = require('../pkg-node/ts_classify.js');

// Three-class dataset: bottom-left (0), bottom-right (1), top (2)
const threeClassSamples = new Float64Array([
  0, -2.0, 1, -2.0,
  0, -1.5, 1, -1.5,
  0, -2.5, 1, -1.0,
  0,  2.0, 1, -2.0,
  0,  1.5, 1, -1.5,
  0,  2.5, 1, -1.0,
  0,  0.0, 1,  2.0,
  0,  0.5, 1,  2.5,
  0, -0.5, 1,  1.5,
]);
const threeClassLengths = new Uint32Array([2, 2, 2, 2, 2, 2, 2, 2, 2]);
const threeClassLabels = new Int32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);

describe('JsSVM', () => {
  it('classifies linearly separable binary data', () => {
    const svm = new JsSVM();

    const samples = new Float64Array([
      0, -2.0, 1,  0.0,
      0, -1.0, 1,  1.0,
      0, -1.0, 1, -1.0,
      0,  1.0, 1,  0.0,
      0,  2.0, 1,  1.0,
      0,  2.0, 1, -1.0,
    ]);
    const lengths = new Uint32Array([2, 2, 2, 2, 2, 2]);
    const labels = new Float64Array([-1, -1, -1, 1, 1, 1]);

    svm.train(samples, lengths, labels);

    assert.equal(svm.predict(new Float64Array([0, -3.0])), -1.0);
    assert.equal(svm.predict(new Float64Array([0,  3.0])),  1.0);
  });

  it('returns a margin value', () => {
    const svm = new JsSVM();

    const samples = new Float64Array([0, -2.0, 0, 2.0]);
    const lengths = new Uint32Array([1, 1]);
    const labels = new Float64Array([-1, 1]);

    svm.train(samples, lengths, labels);

    const margin = svm.margin(new Float64Array([0, 5.0]));
    assert.ok(margin > 0, `expected positive margin, got ${margin}`);
  });

  it('serializes and deserializes', () => {
    const svm = new JsSVM();

    const samples = new Float64Array([0, -2.0, 0, 2.0]);
    const lengths = new Uint32Array([1, 1]);
    const labels = new Float64Array([-1, 1]);

    svm.train(samples, lengths, labels);

    const bytes = svm.to_bytes();
    const loaded = JsSVM.from_bytes(bytes);

    assert.equal(loaded.predict(new Float64Array([0, -3.0])), -1.0);
    assert.equal(loaded.predict(new Float64Array([0,  3.0])),  1.0);
  });
});

describe('JsOneVsRestSVM', () => {
  it('classifies three classes', () => {
    const svm = new JsOneVsRestSVM();
    svm.train(threeClassSamples, threeClassLengths, threeClassLabels);

    assert.equal(svm.num_classifiers(), 3);
    assert.equal(svm.predict(new Float64Array([0, -3.0, 1, -3.0])), 0);
    assert.equal(svm.predict(new Float64Array([0,  3.0, 1, -3.0])), 1);
    assert.equal(svm.predict(new Float64Array([0,  0.0, 1,  3.0])), 2);
  });

  it('returns margins for all classes', () => {
    const svm = new JsOneVsRestSVM();
    svm.train(threeClassSamples, threeClassLengths, threeClassLabels);

    const margins = svm.margins(new Float64Array([0, -3.0, 1, -3.0]));
    // Flat array: [class0, margin0, class1, margin1, ...]
    assert.equal(margins.length, 6);
  });

  it('serializes and deserializes', () => {
    const svm = new JsOneVsRestSVM();
    svm.train(threeClassSamples, threeClassLengths, threeClassLabels);

    const bytes = svm.to_bytes();
    const loaded = JsOneVsRestSVM.from_bytes(bytes);

    assert.equal(loaded.predict(new Float64Array([0, -3.0, 1, -3.0])), 0);
    assert.equal(loaded.predict(new Float64Array([0,  3.0, 1, -3.0])), 1);
  });
});

describe('JsMulticlassSVM', () => {
  it('classifies three classes', () => {
    const svm = new JsMulticlassSVM();
    svm.train(threeClassSamples, threeClassLengths, threeClassLabels);

    assert.equal(svm.num_classifiers(), 3); // 3 choose 2
    assert.equal(svm.num_classes(), 3);
    assert.equal(svm.predict(new Float64Array([0, -3.0, 1, -3.0])), 0);
    assert.equal(svm.predict(new Float64Array([0,  3.0, 1, -3.0])), 1);
    assert.equal(svm.predict(new Float64Array([0,  0.0, 1,  3.0])), 2);
  });

  it('serializes and deserializes', () => {
    const svm = new JsMulticlassSVM();
    svm.train(threeClassSamples, threeClassLengths, threeClassLabels);

    const bytes = svm.to_bytes();
    const loaded = JsMulticlassSVM.from_bytes(bytes);

    assert.equal(loaded.predict(new Float64Array([0, -3.0, 1, -3.0])), 0);
    assert.equal(loaded.predict(new Float64Array([0,  3.0, 1, -3.0])), 1);
  });
});

describe('JsNearestCentroid', () => {
  it('classifies by cosine similarity to centroids', () => {
    // Each class uses distinct feature indices
    const samples = new Float64Array([
      0, 1.0, 1, 0.5,
      0, 1.1, 1, 0.4,
      100, 1.0, 101, 0.5,
      100, 0.9, 101, 0.6,
      200, 1.0, 201, 0.5,
      200, 1.2, 201, 0.3,
    ]);
    const lengths = new Uint32Array([2, 2, 2, 2, 2, 2]);
    const labels = new Int32Array([0, 0, 1, 1, 2, 2]);

    const nc = new JsNearestCentroid();
    nc.train(samples, lengths, labels);

    assert.equal(nc.num_classes(), 3);
    assert.equal(nc.predict(new Float64Array([0, 1.0, 1, 0.5])), 0);
    assert.equal(nc.predict(new Float64Array([100, 1.0, 101, 0.5])), 1);
    assert.equal(nc.predict(new Float64Array([200, 1.0, 201, 0.5])), 2);
  });

  it('returns similarity scores for all classes', () => {
    const samples = new Float64Array([0, 1.0, 100, 1.0]);
    const lengths = new Uint32Array([1, 1]);
    const labels = new Int32Array([0, 1]);

    const nc = new JsNearestCentroid();
    nc.train(samples, lengths, labels);

    const margins = nc.margins(new Float64Array([0, 1.0]));
    // Flat array: [class0, sim0, class1, sim1]
    assert.equal(margins.length, 4);
  });

  it('serializes and deserializes', () => {
    const samples = new Float64Array([0, 1.0, 100, 1.0]);
    const lengths = new Uint32Array([1, 1]);
    const labels = new Int32Array([0, 1]);

    const nc = new JsNearestCentroid();
    nc.train(samples, lengths, labels);

    const bytes = nc.to_bytes();
    const loaded = JsNearestCentroid.from_bytes(bytes);

    assert.equal(loaded.predict(new Float64Array([0, 1.0])), 0);
    assert.equal(loaded.predict(new Float64Array([100, 1.0])), 1);
  });
});
