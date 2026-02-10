// ABOUTME: Classification library with SVM and Nearest Centroid algorithms.
// ABOUTME: Targets both native Rust and WebAssembly compilation.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Sparse vector represented as sorted (index, value) pairs.
pub type SparseVec = Vec<(usize, f64)>;

/// Training algorithm for SVM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Solver {
    /// Sequential Minimal Optimization - works with kernels, O(N²) for linear
    SMO,
    /// Coordinate Descent - linear only, O(N) per pass, much faster for large data
    #[default]
    CoordinateDescent,
}

/// Configuration for SVM training.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Training algorithm to use
    pub solver: Solver,
    /// Regularization parameter (higher = less regularization)
    pub c: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            solver: Solver::default(),
            c: 1.0,
            max_iter: 1000,
            tol: 0.01,
        }
    }
}

impl TrainConfig {
    /// Create config with SMO solver
    pub fn smo() -> Self {
        TrainConfig {
            solver: Solver::SMO,
            c: 1.0,
            max_iter: 10000,
            tol: 1e-4,
        }
    }

    /// Create config with Coordinate Descent solver
    pub fn coordinate_descent() -> Self {
        TrainConfig {
            solver: Solver::CoordinateDescent,
            c: 1.0,
            max_iter: 1000,
            tol: 0.01,
        }
    }

    /// Set regularization parameter
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

/// Multiclass SVM using one-vs-one strategy.
/// For K classes, trains K(K-1)/2 binary classifiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MulticlassSVM {
    /// Binary classifiers indexed by (class_a, class_b) where class_a < class_b
    classifiers: HashMap<(i32, i32), SVM>,
    /// All unique class labels seen during training
    classes: Vec<i32>,
}

impl MulticlassSVM {
    pub fn new() -> Self {
        MulticlassSVM {
            classifiers: HashMap::new(),
            classes: Vec::new(),
        }
    }

    /// Train multiclass SVM on sparse data with default config.
    pub fn train_sparse(&mut self, samples: &[SparseVec], labels: &[i32]) {
        self.train_sparse_with_config(samples, labels, &TrainConfig::default());
    }

    /// Train multiclass SVM on sparse data with custom config.
    /// Labels are integer class IDs (0, 1, 2, ...).
    /// Training is parallelized across all class pairs.
    pub fn train_sparse_with_config(
        &mut self,
        samples: &[SparseVec],
        labels: &[i32],
        config: &TrainConfig,
    ) {
        // Find all unique classes
        let mut classes: Vec<i32> = labels.iter().copied().collect();
        classes.sort();
        classes.dedup();
        self.classes = classes.clone();

        // Generate all class pairs
        let pairs: Vec<(i32, i32)> = {
            let classes = &classes;
            (0..classes.len())
                .flat_map(|i| {
                    let classes = classes.clone();
                    ((i + 1)..classes.len()).map(move |j| (classes[i], classes[j]))
                })
                .collect()
        };

        // Train classifiers (parallel when feature enabled)
        let config = config.clone();
        let train_pair = |&(class_a, class_b): &(i32, i32)| {
            let mut pair_samples = Vec::new();
            let mut pair_labels = Vec::new();

            for (sample, &label) in samples.iter().zip(labels.iter()) {
                if label == class_a {
                    pair_samples.push(sample.clone());
                    pair_labels.push(-1.0);
                } else if label == class_b {
                    pair_samples.push(sample.clone());
                    pair_labels.push(1.0);
                }
            }

            let mut svm = SVM::new();
            svm.train_sparse_with_config(&pair_samples, &pair_labels, &config);
            ((class_a, class_b), svm)
        };

        #[cfg(feature = "parallel")]
        let trained: Vec<_> = pairs.par_iter().map(train_pair).collect();

        #[cfg(not(feature = "parallel"))]
        let trained: Vec<_> = pairs.iter().map(train_pair).collect();

        self.classifiers = trained.into_iter().collect();
    }

    /// Predict class label for a sparse sample using voting.
    pub fn predict_sparse(&self, sample: &[(usize, f64)]) -> i32 {
        let mut votes: HashMap<i32, usize> = HashMap::new();

        for (&(class_a, class_b), svm) in &self.classifiers {
            let prediction = svm.predict_sparse(sample);
            let winner = if prediction < 0.0 { class_a } else { class_b };
            *votes.entry(winner).or_insert(0) += 1;
        }

        // Return class with most votes
        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(class, _)| class)
            .unwrap_or(0)
    }

    /// Number of binary classifiers
    pub fn num_classifiers(&self) -> usize {
        self.classifiers.len()
    }

    /// Number of classes
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Serialize the model to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize a model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map(|(v, _)| v)
    }
}

impl Default for MulticlassSVM {
    fn default() -> Self {
        Self::new()
    }
}

/// Multiclass SVM using one-vs-rest strategy.
/// For K classes, trains K binary classifiers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OneVsRestSVM {
    /// Binary classifiers, one per class (class vs all others)
    classifiers: Vec<(i32, SVM)>,
}

impl OneVsRestSVM {
    pub fn new() -> Self {
        OneVsRestSVM {
            classifiers: Vec::new(),
        }
    }

    /// Train one-vs-rest SVM on sparse data with default config.
    pub fn train_sparse(&mut self, samples: &[SparseVec], labels: &[i32]) {
        self.train_sparse_with_config(samples, labels, &TrainConfig::default());
    }

    /// Train one-vs-rest SVM on sparse data with custom config.
    /// Labels are integer class IDs (0, 1, 2, ...).
    /// Training is parallelized across all classes.
    pub fn train_sparse_with_config(
        &mut self,
        samples: &[SparseVec],
        labels: &[i32],
        config: &TrainConfig,
    ) {
        // Find all unique classes
        let mut classes: Vec<i32> = labels.iter().copied().collect();
        classes.sort();
        classes.dedup();

        // Train classifiers (parallel when feature enabled)
        let config = config.clone();
        let train_class = |&target_class: &i32| {
            let binary_labels: Vec<f64> = labels
                .iter()
                .map(|&l| if l == target_class { 1.0 } else { -1.0 })
                .collect();

            let mut svm = SVM::new();
            svm.train_sparse_with_config(samples, &binary_labels, &config);
            (target_class, svm)
        };

        #[cfg(feature = "parallel")]
        {
            self.classifiers = classes.par_iter().map(train_class).collect();
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.classifiers = classes.iter().map(train_class).collect();
        }
    }

    /// Predict class label for a sparse sample.
    /// Returns the class whose classifier gives the highest margin.
    pub fn predict_sparse(&self, sample: &[(usize, f64)]) -> i32 {
        self.classifiers
            .iter()
            .map(|(class, svm)| (*class, svm.margin_sparse(sample)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
            .unwrap_or(0)
    }

    /// Get margins (decision values) for all classes.
    /// Returns pairs of (class_id, margin) sorted by class_id.
    pub fn margins_sparse(&self, sample: &[(usize, f64)]) -> Vec<(i32, f64)> {
        let mut margins: Vec<(i32, f64)> = self
            .classifiers
            .iter()
            .map(|(class, svm)| (*class, svm.margin_sparse(sample)))
            .collect();
        margins.sort_by_key(|(class, _)| *class);
        margins
    }

    /// Number of binary classifiers
    pub fn num_classifiers(&self) -> usize {
        self.classifiers.len()
    }

    /// Serialize the model to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize a model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map(|(v, _)| v)
    }
}

impl Default for OneVsRestSVM {
    fn default() -> Self {
        Self::new()
    }
}

/// Nearest Centroid classifier.
/// Computes the centroid (average) of each class and predicts by finding
/// the class whose centroid has highest cosine similarity to the input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NearestCentroid {
    /// Centroids per class: (class_id, normalized centroid vector)
    centroids: Vec<(i32, SparseVec)>,
}

impl NearestCentroid {
    pub fn new() -> Self {
        NearestCentroid {
            centroids: Vec::new(),
        }
    }

    /// Train on sparse data.
    /// Labels are integer class IDs (0, 1, 2, ...).
    pub fn train_sparse(&mut self, samples: &[SparseVec], labels: &[i32]) {
        // Group samples by class
        let mut by_class: HashMap<i32, Vec<&SparseVec>> = HashMap::new();
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            by_class.entry(label).or_default().push(sample);
        }

        // Compute centroid for each class
        self.centroids.clear();
        let mut classes: Vec<i32> = by_class.keys().copied().collect();
        classes.sort();

        for class in classes {
            let class_samples = &by_class[&class];
            let centroid = Self::compute_centroid(class_samples);
            self.centroids.push((class, centroid));
        }
    }

    fn compute_centroid(samples: &[&SparseVec]) -> SparseVec {
        if samples.is_empty() {
            return Vec::new();
        }

        // Sum all features across samples
        let mut sums: HashMap<usize, f64> = HashMap::new();
        for sample in samples {
            for &(idx, val) in sample.iter() {
                *sums.entry(idx).or_default() += val;
            }
        }

        // Average and normalize
        let n = samples.len() as f64;
        let mut centroid: SparseVec = sums
            .into_iter()
            .map(|(idx, sum)| (idx, sum / n))
            .collect();
        centroid.sort_by_key(|(idx, _)| *idx);

        // L2 normalize
        let magnitude: f64 = centroid.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for (_, v) in centroid.iter_mut() {
                *v /= magnitude;
            }
        }

        centroid
    }

    /// Predict class for a sparse sample.
    /// Returns the class whose centroid has highest cosine similarity.
    pub fn predict_sparse(&self, sample: &[(usize, f64)]) -> i32 {
        self.margins_sparse(sample)
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
            .unwrap_or(0)
    }

    /// Get cosine similarities to all class centroids.
    pub fn margins_sparse(&self, sample: &[(usize, f64)]) -> Vec<(i32, f64)> {
        // Normalize input sample
        let magnitude: f64 = sample.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        let norm_factor = if magnitude > 0.0 { 1.0 / magnitude } else { 0.0 };

        self.centroids
            .iter()
            .map(|(class, centroid)| {
                let similarity = dot_sparse(sample, centroid) * norm_factor;
                (*class, similarity)
            })
            .collect()
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.centroids.len()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map(|(v, _)| v)
    }
}

impl Default for NearestCentroid {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary linear SVM classifier trained using the SMO algorithm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SVM {
    /// Weight vector (sparse representation)
    w: SparseVec,
    /// Bias term
    b: f64,
}

impl SVM {
    pub fn new() -> Self {
        SVM {
            w: Vec::new(),
            b: 0.0,
        }
    }

    /// Train the SVM using the Simplified SMO algorithm.
    ///
    /// samples: training data, each inner vec is a feature vector
    /// labels: class labels, must be -1.0 or 1.0
    pub fn train(&mut self, samples: &[Vec<f64>], labels: &[f64]) {
        let n = samples.len();

        // SMO parameters
        let c = 1.0;          // regularization parameter
        let tol = 1e-4;       // numerical tolerance
        let max_passes = 10;  // passes without alpha change before stopping
        let max_iter = 10000;

        // Initialize alphas to zero
        let mut alpha = vec![0.0; n];
        self.b = 0.0;

        let mut passes = 0;
        let mut iter = 0;

        while passes < max_passes && iter < max_iter {
            let mut num_changed = 0;

            for i in 0..n {
                // Compute error for sample i
                let ei = self.margin_with_alpha(samples, labels, &alpha, i) - labels[i];

                // Check if alpha[i] violates KKT conditions
                if (labels[i] * ei < -tol && alpha[i] < c)
                    || (labels[i] * ei > tol && alpha[i] > 0.0)
                {
                    // Select j randomly, j != i
                    let mut j = rand_usize(n);
                    while j == i {
                        j = rand_usize(n);
                    }

                    let ej = self.margin_with_alpha(samples, labels, &alpha, j) - labels[j];

                    let ai_old = alpha[i];
                    let aj_old = alpha[j];

                    // Compute bounds L and H
                    let (l, h) = if labels[i] != labels[j] {
                        (
                            f64::max(0.0, alpha[j] - alpha[i]),
                            f64::min(c, c + alpha[j] - alpha[i]),
                        )
                    } else {
                        (
                            f64::max(0.0, alpha[i] + alpha[j] - c),
                            f64::min(c, alpha[i] + alpha[j]),
                        )
                    };

                    if (l - h).abs() < 1e-4 {
                        continue;
                    }

                    // Compute eta (second derivative of objective along constraint)
                    let eta = 2.0 * dot(&samples[i], &samples[j])
                        - dot(&samples[i], &samples[i])
                        - dot(&samples[j], &samples[j]);

                    if eta >= 0.0 {
                        continue;
                    }

                    // Update alpha[j]
                    alpha[j] = aj_old - labels[j] * (ei - ej) / eta;
                    alpha[j] = alpha[j].clamp(l, h);

                    if (alpha[j] - aj_old).abs() < 1e-4 {
                        continue;
                    }

                    // Update alpha[i]
                    alpha[i] = ai_old + labels[i] * labels[j] * (aj_old - alpha[j]);

                    // Update bias
                    let b1 = self.b - ei
                        - labels[i] * (alpha[i] - ai_old) * dot(&samples[i], &samples[i])
                        - labels[j] * (alpha[j] - aj_old) * dot(&samples[i], &samples[j]);
                    let b2 = self.b - ej
                        - labels[i] * (alpha[i] - ai_old) * dot(&samples[i], &samples[j])
                        - labels[j] * (alpha[j] - aj_old) * dot(&samples[j], &samples[j]);

                    if alpha[i] > 0.0 && alpha[i] < c {
                        self.b = b1;
                    } else if alpha[j] > 0.0 && alpha[j] < c {
                        self.b = b2;
                    } else {
                        self.b = (b1 + b2) / 2.0;
                    }

                    num_changed += 1;
                }
            }

            iter += 1;
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        // Compute weight vector w = sum(alpha_i * y_i * x_i)
        // Convert dense samples to sparse for weight accumulation
        self.w = Vec::new();
        for i in 0..n {
            let sparse_sample: SparseVec = samples[i]
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v.abs() > 1e-10)
                .map(|(idx, &v)| (idx, v))
                .collect();
            add_sparse(&mut self.w, &sparse_sample, alpha[i] * labels[i]);
        }
    }

    /// Compute margin for sample at index, using current alpha values during training.
    fn margin_with_alpha(
        &self,
        samples: &[Vec<f64>],
        labels: &[f64],
        alpha: &[f64],
        idx: usize,
    ) -> f64 {
        let mut f = self.b;
        for i in 0..samples.len() {
            f += alpha[i] * labels[i] * dot(&samples[i], &samples[idx]);
        }
        f
    }

    /// Predict the class label for a dense sample.
    pub fn predict(&self, sample: &[f64]) -> f64 {
        // Convert dense to sparse for dot product
        let sparse: SparseVec = sample
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v.abs() > 1e-10)
            .map(|(i, &v)| (i, v))
            .collect();
        self.predict_sparse(&sparse)
    }

    /// Train the SVM using sparse vectors with the default config.
    pub fn train_sparse(&mut self, samples: &[SparseVec], labels: &[f64]) {
        self.train_sparse_with_config(samples, labels, &TrainConfig::default());
    }

    /// Train with full configuration.
    pub fn train_sparse_with_config(
        &mut self,
        samples: &[SparseVec],
        labels: &[f64],
        config: &TrainConfig,
    ) {
        match config.solver {
            Solver::SMO => self.train_sparse_smo(samples, labels, config),
            Solver::CoordinateDescent => self.train_sparse_cd(samples, labels, config),
        }
    }

    /// Compute margin for sparse sample at index during training.
    fn margin_sparse_with_alpha(
        &self,
        samples: &[SparseVec],
        labels: &[f64],
        alpha: &[f64],
        idx: usize,
    ) -> f64 {
        let mut f = self.b;
        for i in 0..samples.len() {
            f += alpha[i] * labels[i] * dot_sparse(&samples[i], &samples[idx]);
        }
        f
    }

    /// Get the raw margin (decision value) for a sparse sample.
    pub fn margin_sparse(&self, sample: &[(usize, f64)]) -> f64 {
        dot_sparse(&self.w, sample) + self.b
    }

    /// Predict the class label for a sparse sample.
    pub fn predict_sparse(&self, sample: &[(usize, f64)]) -> f64 {
        if self.margin_sparse(sample) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Train with a specific solver algorithm (convenience method).
    pub fn train_sparse_with_solver(
        &mut self,
        samples: &[SparseVec],
        labels: &[f64],
        solver: Solver,
    ) {
        let config = match solver {
            Solver::SMO => TrainConfig::smo(),
            Solver::CoordinateDescent => TrainConfig::coordinate_descent(),
        };
        self.train_sparse_with_config(samples, labels, &config);
    }

    /// Train using SMO algorithm.
    fn train_sparse_smo(&mut self, samples: &[SparseVec], labels: &[f64], config: &TrainConfig) {
        let n = samples.len();
        let c = config.c;
        let tol = config.tol;
        let max_passes = 10;
        let max_iter = config.max_iter;

        let mut alpha = vec![0.0; n];
        self.b = 0.0;
        self.w = Vec::new();

        let mut passes = 0;
        let mut iter = 0;

        while passes < max_passes && iter < max_iter {
            let mut num_changed = 0;

            for i in 0..n {
                let ei = self.margin_sparse_with_alpha(samples, labels, &alpha, i) - labels[i];

                if (labels[i] * ei < -tol && alpha[i] < c)
                    || (labels[i] * ei > tol && alpha[i] > 0.0)
                {
                    let mut j = rand_usize(n);
                    while j == i {
                        j = rand_usize(n);
                    }

                    let ej = self.margin_sparse_with_alpha(samples, labels, &alpha, j) - labels[j];

                    let ai_old = alpha[i];
                    let aj_old = alpha[j];

                    let (l, h) = if labels[i] != labels[j] {
                        (
                            f64::max(0.0, alpha[j] - alpha[i]),
                            f64::min(c, c + alpha[j] - alpha[i]),
                        )
                    } else {
                        (
                            f64::max(0.0, alpha[i] + alpha[j] - c),
                            f64::min(c, alpha[i] + alpha[j]),
                        )
                    };

                    if (l - h).abs() < 1e-4 {
                        continue;
                    }

                    let eta = 2.0 * dot_sparse(&samples[i], &samples[j])
                        - dot_sparse(&samples[i], &samples[i])
                        - dot_sparse(&samples[j], &samples[j]);

                    if eta >= 0.0 {
                        continue;
                    }

                    alpha[j] = aj_old - labels[j] * (ei - ej) / eta;
                    alpha[j] = alpha[j].clamp(l, h);

                    if (alpha[j] - aj_old).abs() < 1e-4 {
                        continue;
                    }

                    alpha[i] = ai_old + labels[i] * labels[j] * (aj_old - alpha[j]);

                    let b1 = self.b - ei
                        - labels[i] * (alpha[i] - ai_old) * dot_sparse(&samples[i], &samples[i])
                        - labels[j] * (alpha[j] - aj_old) * dot_sparse(&samples[i], &samples[j]);
                    let b2 = self.b - ej
                        - labels[i] * (alpha[i] - ai_old) * dot_sparse(&samples[i], &samples[j])
                        - labels[j] * (alpha[j] - aj_old) * dot_sparse(&samples[j], &samples[j]);

                    if alpha[i] > 0.0 && alpha[i] < c {
                        self.b = b1;
                    } else if alpha[j] > 0.0 && alpha[j] < c {
                        self.b = b2;
                    } else {
                        self.b = (b1 + b2) / 2.0;
                    }

                    num_changed += 1;
                }
            }

            iter += 1;
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        self.w = Vec::new();
        for i in 0..n {
            add_sparse(&mut self.w, &samples[i], alpha[i] * labels[i]);
        }
    }

    /// Train using Coordinate Descent (primal, L2-regularized L2-loss SVM).
    /// Much faster than SMO for linear SVM on large datasets.
    fn train_sparse_cd(&mut self, samples: &[SparseVec], labels: &[f64], config: &TrainConfig) {
        let n = samples.len();
        if n == 0 {
            return;
        }

        // Find max feature index to size our weight vector
        let max_idx = samples
            .iter()
            .flat_map(|s| s.iter().map(|(idx, _)| *idx))
            .max()
            .unwrap_or(0);

        // Parameters from config
        let c = config.c;
        let eps = config.tol;
        let max_iter = config.max_iter;

        // Use dense weights internally for coordinate descent (faster updates)
        let mut w = vec![0.0; max_idx + 1];
        self.b = 0.0;

        // Precompute squared norms of samples (for diagonal of Q matrix)
        let diag: Vec<f64> = samples.iter().map(|s| dot_sparse(s, s)).collect();

        // Upper bound on alpha: for L2-loss SVM, it's infinity, but we use C for stability
        let upper_bound = f64::MAX;

        // Alpha values (dual variables)
        let mut alpha = vec![0.0; n];

        for _ in 0..max_iter {
            let mut max_violation = 0.0f64;

            for i in 0..n {
                let y_i = labels[i];

                // Compute current prediction: w · x_i
                let mut wx: f64 = 0.0;
                for &(idx, val) in &samples[i] {
                    wx += w[idx] * val;
                }

                // Gradient of the dual objective for alpha[i]
                // For L2-loss: G = y_i * wx - 1 + alpha[i] / (2*C)
                let g = y_i * wx - 1.0 + alpha[i] / (2.0 * c);

                // Project gradient
                let pg = if alpha[i] == 0.0 {
                    g.min(0.0)
                } else if alpha[i] == upper_bound {
                    g.max(0.0)
                } else {
                    g
                };

                max_violation = max_violation.max(pg.abs());

                if pg.abs() < 1e-12 {
                    continue;
                }

                // Coordinate descent update
                // Q_ii = x_i · x_i + 1/(2C) for L2-loss
                let q_ii = diag[i] + 1.0 / (2.0 * c);
                let alpha_old = alpha[i];
                alpha[i] = (alpha[i] - g / q_ii).max(0.0).min(upper_bound);
                let d = alpha[i] - alpha_old;

                if d.abs() < 1e-12 {
                    continue;
                }

                // Update weights: w += d * y_i * x_i
                for &(idx, val) in &samples[i] {
                    w[idx] += d * y_i * val;
                }
            }

            if max_violation < eps {
                break;
            }
        }

        // Convert dense weights to sparse
        self.w = w
            .into_iter()
            .enumerate()
            .filter(|&(_, v)| v.abs() > 1e-10)
            .collect();

        // Compute bias using support vectors (samples with 0 < alpha < C)
        // For simplicity, use average over all correctly-margined samples
        let mut b_sum = 0.0;
        let mut b_count = 0;
        for i in 0..n {
            if alpha[i] > 1e-8 {
                let wx: f64 = samples[i].iter().map(|&(idx, v)| {
                    self.w.binary_search_by_key(&idx, |&(i, _)| i)
                        .map(|j| self.w[j].1 * v)
                        .unwrap_or(0.0)
                }).sum();
                b_sum += labels[i] - wx;
                b_count += 1;
            }
        }
        self.b = if b_count > 0 { b_sum / b_count as f64 } else { 0.0 };
    }

    /// Serialize the model to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
    }

    /// Deserialize a model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map(|(v, _)| v)
    }
}

impl Default for SVM {
    fn default() -> Self {
        Self::new()
    }
}

/// Dot product of two dense vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product of two sparse vectors (must be sorted by index).
fn dot_sparse(a: &[(usize, f64)], b: &[(usize, f64)]) -> f64 {
    let mut result = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    result
}

/// Add a scaled sparse vector to another: dest += scale * src
fn add_sparse(dest: &mut SparseVec, src: &[(usize, f64)], scale: f64) {
    use std::collections::BTreeMap;

    // Convert dest to map for easy updates
    let mut map: BTreeMap<usize, f64> = dest.iter().copied().collect();

    for &(idx, val) in src {
        *map.entry(idx).or_insert(0.0) += scale * val;
    }

    // Convert back, filtering near-zero values
    *dest = map
        .into_iter()
        .filter(|&(_, v)| v.abs() > 1e-10)
        .collect();
}

/// Simple random number generator (not cryptographically secure).
fn rand_usize(max: usize) -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
        }
        // LCG
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        (SEED as usize) % max
    }
}

// ============================================================================
// WASM Bindings - JS-friendly API
// ============================================================================

/// JS-friendly wrapper for binary SVM.
/// Uses flat arrays for sparse vectors: [idx0, val0, idx1, val1, ...]
#[wasm_bindgen]
pub struct JsSVM {
    inner: SVM,
}

#[wasm_bindgen]
impl JsSVM {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsSVM { inner: SVM::new() }
    }

    /// Train on sparse data.
    /// samples_flat: flattened sparse vectors [idx, val, idx, val, ...]
    /// sample_lengths: number of (idx, val) pairs per sample
    /// labels: -1.0 or 1.0 for each sample
    #[wasm_bindgen]
    pub fn train(&mut self, samples_flat: &[f64], sample_lengths: &[usize], labels: &[f64]) {
        let samples = unflatten_samples(samples_flat, sample_lengths);
        self.inner.train_sparse(&samples, labels);
    }

    /// Train with custom parameters.
    #[wasm_bindgen]
    pub fn train_with_config(
        &mut self,
        samples_flat: &[f64],
        sample_lengths: &[usize],
        labels: &[f64],
        use_smo: bool,
        c: f64,
        max_iter: usize,
        tol: f64,
    ) {
        let samples = unflatten_samples(samples_flat, sample_lengths);
        let config = TrainConfig {
            solver: if use_smo { Solver::SMO } else { Solver::CoordinateDescent },
            c,
            max_iter,
            tol,
        };
        self.inner.train_sparse_with_config(&samples, labels, &config);
    }

    /// Predict class for a single sparse sample.
    /// sample_flat: [idx, val, idx, val, ...]
    #[wasm_bindgen]
    pub fn predict(&self, sample_flat: &[f64]) -> f64 {
        let sample = unflatten_single(sample_flat);
        self.inner.predict_sparse(&sample)
    }

    /// Get the margin (decision value) for a sample.
    #[wasm_bindgen]
    pub fn margin(&self, sample_flat: &[f64]) -> f64 {
        let sample = unflatten_single(sample_flat);
        self.inner.margin_sparse(&sample)
    }

    /// Serialize to bytes.
    #[wasm_bindgen]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.inner.to_bytes().map_err(|e| JsError::new(&e.to_string()))
    }

    /// Deserialize from bytes.
    #[wasm_bindgen]
    pub fn from_bytes(bytes: &[u8]) -> Result<JsSVM, JsError> {
        SVM::from_bytes(bytes)
            .map(|inner| JsSVM { inner })
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

/// JS-friendly wrapper for One-vs-Rest multiclass SVM.
#[wasm_bindgen]
pub struct JsOneVsRestSVM {
    inner: OneVsRestSVM,
}

#[wasm_bindgen]
impl JsOneVsRestSVM {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsOneVsRestSVM { inner: OneVsRestSVM::new() }
    }

    /// Train on sparse data.
    /// labels: integer class IDs (0, 1, 2, ...)
    #[wasm_bindgen]
    pub fn train(&mut self, samples_flat: &[f64], sample_lengths: &[usize], labels: &[i32]) {
        let samples = unflatten_samples(samples_flat, sample_lengths);
        self.inner.train_sparse(&samples, labels);
    }

    /// Train with custom parameters.
    #[wasm_bindgen]
    pub fn train_with_config(
        &mut self,
        samples_flat: &[f64],
        sample_lengths: &[usize],
        labels: &[i32],
        use_smo: bool,
        c: f64,
        max_iter: usize,
        tol: f64,
    ) {
        let samples = unflatten_samples(samples_flat, sample_lengths);
        let config = TrainConfig {
            solver: if use_smo { Solver::SMO } else { Solver::CoordinateDescent },
            c,
            max_iter,
            tol,
        };
        self.inner.train_sparse_with_config(&samples, labels, &config);
    }

    /// Predict class for a single sparse sample.
    #[wasm_bindgen]
    pub fn predict(&self, sample_flat: &[f64]) -> i32 {
        let sample = unflatten_single(sample_flat);
        self.inner.predict_sparse(&sample)
    }

    /// Get margins for all classes.
    /// Returns flat array: [class0, margin0, class1, margin1, ...]
    #[wasm_bindgen]
    pub fn margins(&self, sample_flat: &[f64]) -> Vec<f64> {
        let sample = unflatten_single(sample_flat);
        let margins = self.inner.margins_sparse(&sample);
        let mut result = Vec::with_capacity(margins.len() * 2);
        for (class, margin) in margins {
            result.push(class as f64);
            result.push(margin);
        }
        result
    }

    /// Number of classifiers.
    #[wasm_bindgen]
    pub fn num_classifiers(&self) -> usize {
        self.inner.num_classifiers()
    }

    /// Serialize to bytes.
    #[wasm_bindgen]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.inner.to_bytes().map_err(|e| JsError::new(&e.to_string()))
    }

    /// Deserialize from bytes.
    #[wasm_bindgen]
    pub fn from_bytes(bytes: &[u8]) -> Result<JsOneVsRestSVM, JsError> {
        OneVsRestSVM::from_bytes(bytes)
            .map(|inner| JsOneVsRestSVM { inner })
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

/// JS-friendly wrapper for Nearest Centroid classifier.
#[wasm_bindgen]
pub struct JsNearestCentroid {
    inner: NearestCentroid,
}

#[wasm_bindgen]
impl JsNearestCentroid {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsNearestCentroid {
            inner: NearestCentroid::new(),
        }
    }

    /// Train on sparse data.
    /// samples_flat: flattened sparse vectors [idx, val, idx, val, ...]
    /// sample_lengths: number of (idx, val) pairs per sample
    /// labels: integer class IDs (0, 1, 2, ...)
    #[wasm_bindgen]
    pub fn train(&mut self, samples_flat: &[f64], sample_lengths: &[usize], labels: &[i32]) {
        let samples = unflatten_samples(samples_flat, sample_lengths);
        self.inner.train_sparse(&samples, labels);
    }

    /// Predict class for a single sparse sample.
    #[wasm_bindgen]
    pub fn predict(&self, sample_flat: &[f64]) -> i32 {
        let sample = unflatten_single(sample_flat);
        self.inner.predict_sparse(&sample)
    }

    /// Get cosine similarities to all class centroids.
    /// Returns flat array: [class0, similarity0, class1, similarity1, ...]
    #[wasm_bindgen]
    pub fn margins(&self, sample_flat: &[f64]) -> Vec<f64> {
        let sample = unflatten_single(sample_flat);
        let margins = self.inner.margins_sparse(&sample);
        let mut result = Vec::with_capacity(margins.len() * 2);
        for (class, margin) in margins {
            result.push(class as f64);
            result.push(margin);
        }
        result
    }

    /// Number of classes.
    #[wasm_bindgen]
    pub fn num_classes(&self) -> usize {
        self.inner.num_classes()
    }

    /// Serialize to bytes.
    #[wasm_bindgen]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.inner
            .to_bytes()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Deserialize from bytes.
    #[wasm_bindgen]
    pub fn from_bytes(bytes: &[u8]) -> Result<JsNearestCentroid, JsError> {
        NearestCentroid::from_bytes(bytes)
            .map(|inner| JsNearestCentroid { inner })
            .map_err(|e| JsError::new(&e.to_string()))
    }
}

/// Convert flat array to sparse vectors.
/// flat: [idx0, val0, idx1, val1, ...]
/// lengths: number of (idx, val) pairs per sample
fn unflatten_samples(flat: &[f64], lengths: &[usize]) -> Vec<SparseVec> {
    let mut samples = Vec::with_capacity(lengths.len());
    let mut offset = 0;

    for &len in lengths {
        let mut sample = Vec::with_capacity(len);
        for _ in 0..len {
            let idx = flat[offset] as usize;
            let val = flat[offset + 1];
            sample.push((idx, val));
            offset += 2;
        }
        samples.push(sample);
    }

    samples
}

/// Convert flat array to single sparse vector.
fn unflatten_single(flat: &[f64]) -> SparseVec {
    let mut sample = Vec::with_capacity(flat.len() / 2);
    for i in (0..flat.len()).step_by(2) {
        let idx = flat[i] as usize;
        let val = flat[i + 1];
        sample.push((idx, val));
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_linear_separable() {
        // Simple linearly separable dataset
        // Left side: class -1, Right side: class +1
        let samples = vec![
            vec![-2.0, 0.0],
            vec![-1.0, 1.0],
            vec![-1.0, -1.0],
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![2.0, -1.0],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train(&samples, &labels);

        // Should classify new points correctly
        assert_eq!(svm.predict(&[-3.0, 0.0]), -1.0);
        assert_eq!(svm.predict(&[3.0, 0.0]), 1.0);
    }

    #[test]
    fn test_higher_dimensional() {
        // 4D linearly separable: first dimension determines class
        let samples = vec![
            vec![-1.0, 0.5, 0.2, -0.3],
            vec![-2.0, -0.5, 0.8, 0.1],
            vec![-0.5, 0.1, -0.4, 0.7],
            vec![1.0, 0.3, -0.2, 0.5],
            vec![2.0, -0.4, 0.6, -0.1],
            vec![0.5, 0.7, 0.1, -0.5],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train(&samples, &labels);

        // Points with negative first coord should be -1
        assert_eq!(svm.predict(&[-1.5, 0.0, 0.0, 0.0]), -1.0);
        // Points with positive first coord should be +1
        assert_eq!(svm.predict(&[1.5, 0.0, 0.0, 0.0]), 1.0);
    }

    #[test]
    fn test_classifies_training_data_correctly() {
        let samples = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 3.0],
            vec![4.0, 3.0],
            vec![3.0, 4.0],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train(&samples, &labels);

        // Should correctly classify all training points
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            assert_eq!(svm.predict(sample), label);
        }
    }

    #[test]
    fn test_sparse_binary_linear_separable() {
        // Same as test_binary_linear_separable but with sparse vectors
        // Sparse format: Vec<(index, value)>
        // Imagine a 1000-dimensional space where only indices 0 and 1 are non-zero
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, 0.0)],
            vec![(0, -1.0), (1, 1.0)],
            vec![(0, -1.0), (1, -1.0)],
            vec![(0, 1.0), (1, 0.0)],
            vec![(0, 2.0), (1, 1.0)],
            vec![(0, 2.0), (1, -1.0)],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train_sparse(&samples, &labels);

        // Should classify new sparse points correctly
        assert_eq!(svm.predict_sparse(&[(0, -3.0)]), -1.0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0)]), 1.0);
    }

    #[test]
    fn test_sparse_high_dimensional() {
        // Sparse vectors in a very high-dimensional space
        // Only a few features are non-zero per sample
        let samples: Vec<SparseVec> = vec![
            vec![(5, 1.0), (100, -2.0), (999, 0.5)],   // class -1
            vec![(5, 0.8), (100, -1.5), (999, 0.3)],   // class -1
            vec![(5, -1.0), (100, 2.0), (999, -0.5)],  // class +1
            vec![(5, -0.8), (100, 1.5), (999, -0.3)],  // class +1
        ];
        let labels = vec![-1.0, -1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train_sparse(&samples, &labels);

        // New points following the pattern
        assert_eq!(svm.predict_sparse(&[(5, 1.2), (100, -2.5)]), -1.0);
        assert_eq!(svm.predict_sparse(&[(5, -1.2), (100, 2.5)]), 1.0);
    }

    #[test]
    fn test_multiclass_three_classes() {
        // Three classes separated in 2D space
        // Class 0: bottom-left, Class 1: bottom-right, Class 2: top
        let samples: Vec<SparseVec> = vec![
            // Class 0 (bottom-left)
            vec![(0, -2.0), (1, -2.0)],
            vec![(0, -1.5), (1, -1.5)],
            vec![(0, -2.5), (1, -1.0)],
            // Class 1 (bottom-right)
            vec![(0, 2.0), (1, -2.0)],
            vec![(0, 1.5), (1, -1.5)],
            vec![(0, 2.5), (1, -1.0)],
            // Class 2 (top)
            vec![(0, 0.0), (1, 2.0)],
            vec![(0, 0.5), (1, 2.5)],
            vec![(0, -0.5), (1, 1.5)],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut svm = MulticlassSVM::new();
        svm.train_sparse(&samples, &labels);

        // Should have 3 binary classifiers for 3 classes
        assert_eq!(svm.num_classifiers(), 3);
        assert_eq!(svm.num_classes(), 3);

        // Test predictions
        assert_eq!(svm.predict_sparse(&[(0, -3.0), (1, -3.0)]), 0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0), (1, -3.0)]), 1);
        assert_eq!(svm.predict_sparse(&[(0, 0.0), (1, 3.0)]), 2);
    }

    #[test]
    fn test_multiclass_classifies_training_data() {
        // 5 classes with 4 samples each
        let mut samples = Vec::new();
        let mut labels = Vec::new();

        for class in 0..5 {
            for i in 0..4 {
                // Each class has samples clustered around (class * 10, class * 10)
                let x = (class as f64) * 10.0 + (i as f64) * 0.5;
                let y = (class as f64) * 10.0 + (i as f64) * 0.3;
                samples.push(vec![(0, x), (1, y)]);
                labels.push(class);
            }
        }

        let mut svm = MulticlassSVM::new();
        svm.train_sparse(&samples, &labels);

        // Should correctly classify all training points
        let mut correct = 0;
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            if svm.predict_sparse(sample) == label {
                correct += 1;
            }
        }
        // Allow some slack, but most should be correct
        assert!(correct >= 18, "Expected at least 18/20 correct, got {}", correct);
    }

    #[test]
    fn test_one_vs_rest_three_classes() {
        // Same test as test_multiclass_three_classes but with OvR
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, -2.0)],
            vec![(0, -1.5), (1, -1.5)],
            vec![(0, -2.5), (1, -1.0)],
            vec![(0, 2.0), (1, -2.0)],
            vec![(0, 1.5), (1, -1.5)],
            vec![(0, 2.5), (1, -1.0)],
            vec![(0, 0.0), (1, 2.0)],
            vec![(0, 0.5), (1, 2.5)],
            vec![(0, -0.5), (1, 1.5)],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut svm = OneVsRestSVM::new();
        svm.train_sparse(&samples, &labels);

        // Should have 3 classifiers for 3 classes
        assert_eq!(svm.num_classifiers(), 3);

        // Test predictions
        assert_eq!(svm.predict_sparse(&[(0, -3.0), (1, -3.0)]), 0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0), (1, -3.0)]), 1);
        assert_eq!(svm.predict_sparse(&[(0, 0.0), (1, 3.0)]), 2);
    }

    #[test]
    fn test_one_vs_rest_classifies_training_data() {
        // Test OvR with sparse features - each class uses distinct feature indices
        // This mirrors how text classification works (each word = feature)
        let mut samples = Vec::new();
        let mut labels = Vec::new();

        for class in 0..5 {
            for i in 0..4 {
                // Each class uses its own feature index range
                // Class 0: features 0-9, Class 1: features 100-109, etc.
                let base_idx = class as usize * 100;
                let sample = vec![
                    (base_idx, 1.0 + i as f64 * 0.1),
                    (base_idx + 1, 0.5 + i as f64 * 0.05),
                ];
                samples.push(sample);
                labels.push(class);
            }
        }

        let mut svm = OneVsRestSVM::new();
        svm.train_sparse(&samples, &labels);

        let mut correct = 0;
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            if svm.predict_sparse(sample) == label {
                correct += 1;
            }
        }
        assert!(correct >= 18, "Expected at least 18/20 correct, got {}", correct);
    }

    #[test]
    fn test_coordinate_descent_basic() {
        // Same test as binary linear separable but using coordinate descent
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, 0.0)],
            vec![(0, -1.0), (1, 1.0)],
            vec![(0, -1.0), (1, -1.0)],
            vec![(0, 1.0), (1, 0.0)],
            vec![(0, 2.0), (1, 1.0)],
            vec![(0, 2.0), (1, -1.0)],
        ];
        let labels = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train_sparse_with_solver(&samples, &labels, Solver::CoordinateDescent);

        assert_eq!(svm.predict_sparse(&[(0, -3.0)]), -1.0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0)]), 1.0);
    }

    #[test]
    fn test_smo_explicit() {
        // Explicit SMO solver selection
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, 0.0)],
            vec![(0, -1.0), (1, 1.0)],
            vec![(0, 1.0), (1, 0.0)],
            vec![(0, 2.0), (1, 1.0)],
        ];
        let labels = vec![-1.0, -1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train_sparse_with_solver(&samples, &labels, Solver::SMO);

        assert_eq!(svm.predict_sparse(&[(0, -3.0)]), -1.0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0)]), 1.0);
    }

    #[test]
    fn test_config_custom_params() {
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, 0.0)],
            vec![(0, -1.0), (1, 1.0)],
            vec![(0, 1.0), (1, 0.0)],
            vec![(0, 2.0), (1, 1.0)],
        ];
        let labels = vec![-1.0, -1.0, 1.0, 1.0];

        // Use custom config with builder pattern
        let config = TrainConfig::coordinate_descent()
            .with_c(0.5)
            .with_max_iter(500)
            .with_tol(0.001);

        let mut svm = SVM::new();
        svm.train_sparse_with_config(&samples, &labels, &config);

        assert_eq!(svm.predict_sparse(&[(0, -3.0)]), -1.0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0)]), 1.0);
    }

    #[test]
    fn test_multiclass_with_config() {
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, -2.0)],
            vec![(0, -1.5), (1, -1.5)],
            vec![(0, 2.0), (1, -2.0)],
            vec![(0, 1.5), (1, -1.5)],
            vec![(0, 0.0), (1, 2.0)],
            vec![(0, 0.5), (1, 2.5)],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];

        let config = TrainConfig::coordinate_descent().with_max_iter(200);

        let mut svm = OneVsRestSVM::new();
        svm.train_sparse_with_config(&samples, &labels, &config);

        assert_eq!(svm.predict_sparse(&[(0, -3.0), (1, -3.0)]), 0);
        assert_eq!(svm.predict_sparse(&[(0, 3.0), (1, -3.0)]), 1);
        assert_eq!(svm.predict_sparse(&[(0, 0.0), (1, 3.0)]), 2);
    }

    #[test]
    fn test_binary_svm_serialization() {
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, 0.0)],
            vec![(0, -1.0), (1, 1.0)],
            vec![(0, 1.0), (1, 0.0)],
            vec![(0, 2.0), (1, 1.0)],
        ];
        let labels = vec![-1.0, -1.0, 1.0, 1.0];

        let mut svm = SVM::new();
        svm.train_sparse(&samples, &labels);

        // Serialize to bytes
        let bytes = svm.to_bytes().unwrap();

        // Deserialize
        let loaded = SVM::from_bytes(&bytes).unwrap();

        // Should predict the same
        assert_eq!(loaded.predict_sparse(&[(0, -3.0)]), -1.0);
        assert_eq!(loaded.predict_sparse(&[(0, 3.0)]), 1.0);
    }

    #[test]
    fn test_multiclass_serialization() {
        let samples: Vec<SparseVec> = vec![
            vec![(0, -2.0), (1, -2.0)],
            vec![(0, -1.5), (1, -1.5)],
            vec![(0, 2.0), (1, -2.0)],
            vec![(0, 1.5), (1, -1.5)],
            vec![(0, 0.0), (1, 2.0)],
            vec![(0, 0.5), (1, 2.5)],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];

        let mut svm = OneVsRestSVM::new();
        svm.train_sparse(&samples, &labels);

        // Serialize and deserialize
        let bytes = svm.to_bytes().unwrap();
        let loaded = OneVsRestSVM::from_bytes(&bytes).unwrap();

        // Should predict the same
        assert_eq!(loaded.predict_sparse(&[(0, -3.0), (1, -3.0)]), 0);
        assert_eq!(loaded.predict_sparse(&[(0, 3.0), (1, -3.0)]), 1);
        assert_eq!(loaded.predict_sparse(&[(0, 0.0), (1, 3.0)]), 2);
    }

    #[test]
    fn test_nearest_centroid_basic() {
        // Three clusters with distinct features
        let samples = vec![
            vec![(0, 1.0), (1, 0.5)],
            vec![(0, 1.1), (1, 0.4)],
            vec![(100, 1.0), (101, 0.5)],
            vec![(100, 0.9), (101, 0.6)],
            vec![(200, 1.0), (201, 0.5)],
            vec![(200, 1.2), (201, 0.3)],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];

        let mut nc = NearestCentroid::new();
        nc.train_sparse(&samples, &labels);

        assert_eq!(nc.num_classes(), 3);

        // Should classify training data correctly
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            assert_eq!(nc.predict_sparse(sample), label);
        }
    }

    #[test]
    fn test_nearest_centroid_serialization() {
        let samples = vec![
            vec![(0, 1.0)],
            vec![(0, 1.1)],
            vec![(100, 1.0)],
            vec![(100, 0.9)],
        ];
        let labels = vec![0, 0, 1, 1];

        let mut nc = NearestCentroid::new();
        nc.train_sparse(&samples, &labels);

        let bytes = nc.to_bytes().unwrap();
        let loaded = NearestCentroid::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.predict_sparse(&[(0, 1.0)]), 0);
        assert_eq!(loaded.predict_sparse(&[(100, 1.0)]), 1);
    }
}

/// Benchmarks and performance tests
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Generate synthetic sparse data for benchmarking.
    /// Each class gets its own distinct region in feature space.
    fn generate_data(
        num_samples: usize,
        num_classes: i32,
        features_per_sample: usize,
        feature_space: usize,
    ) -> (Vec<SparseVec>, Vec<i32>) {
        let mut samples = Vec::with_capacity(num_samples);
        let mut labels = Vec::with_capacity(num_samples);

        // Each class gets a dedicated range in feature space
        let features_per_class = feature_space / num_classes as usize;

        for i in 0..num_samples {
            let class = (i as i32) % num_classes;
            labels.push(class);

            // Each class uses features in its own range
            let base_idx = class as usize * features_per_class;
            let mut sample = Vec::with_capacity(features_per_sample);
            for j in 0..features_per_sample {
                let idx = base_idx + (j % features_per_class);
                // Value varies slightly within class but is class-specific
                let val = 1.0 + (j as f64) * 0.01 + (i as f64) * 0.001;
                sample.push((idx, val));
            }
            // Remove duplicates (keep last) and sort
            sample.sort_by_key(|&(idx, _)| idx);
            sample.dedup_by_key(|(idx, _)| *idx);
            samples.push(sample);
        }

        (samples, labels)
    }

    #[test]
    fn bench_binary_training() {
        let (samples, labels_i32) = generate_data(1000, 2, 50, 10000);
        let labels: Vec<f64> = labels_i32.iter().map(|&l| if l == 0 { -1.0 } else { 1.0 }).collect();

        let start = Instant::now();
        let mut svm = SVM::new();
        svm.train_sparse(&samples, &labels);
        let elapsed = start.elapsed();

        println!("Binary SVM training (1000 samples, 50 features): {:?}", elapsed);
        assert!(elapsed.as_secs() < 10, "Training took too long: {:?}", elapsed);
    }

    #[test]
    fn bench_binary_prediction() {
        let (samples, labels_i32) = generate_data(100, 2, 50, 10000);
        let labels: Vec<f64> = labels_i32.iter().map(|&l| if l == 0 { -1.0 } else { 1.0 }).collect();

        let mut svm = SVM::new();
        svm.train_sparse(&samples, &labels);

        // Benchmark prediction
        let test_sample = &samples[0];
        let iterations = 10000;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = svm.predict_sparse(test_sample);
        }
        let elapsed = start.elapsed();

        let per_prediction = elapsed / iterations;
        println!("Binary SVM prediction: {:?} per prediction", per_prediction);
        assert!(per_prediction.as_micros() < 100, "Prediction too slow: {:?}", per_prediction);
    }

    #[test]
    fn bench_multiclass_small() {
        // 10 classes - manageable for quick tests
        let num_classes = 10;
        let (samples, labels) = generate_data(200, num_classes, 20, 5000);

        let start = Instant::now();
        let mut svm = MulticlassSVM::new();
        svm.train_sparse(&samples, &labels);
        let train_elapsed = start.elapsed();

        let expected_classifiers = (num_classes * (num_classes - 1) / 2) as usize;
        assert_eq!(svm.num_classifiers(), expected_classifiers);

        println!(
            "Multiclass SVM ({} classes, {} classifiers, {} samples): {:?}",
            num_classes, expected_classifiers, samples.len(), train_elapsed
        );

        // Benchmark prediction
        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = svm.predict_sparse(&samples[0]);
        }
        let predict_elapsed = start.elapsed();
        println!(
            "Multiclass prediction: {:?} per prediction",
            predict_elapsed / iterations
        );
    }

    #[test]
    fn bench_multiclass_medium() {
        // 50 classes = 1225 classifiers
        let num_classes = 50;
        let (samples, labels) = generate_data(500, num_classes, 20, 5000);

        let start = Instant::now();
        let mut svm = MulticlassSVM::new();
        svm.train_sparse(&samples, &labels);
        let train_elapsed = start.elapsed();

        let expected_classifiers = (num_classes * (num_classes - 1) / 2) as usize;
        println!(
            "Multiclass SVM ({} classes, {} classifiers, {} samples): {:?}",
            num_classes, expected_classifiers, samples.len(), train_elapsed
        );

        // Benchmark prediction
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = svm.predict_sparse(&samples[0]);
        }
        let predict_elapsed = start.elapsed();
        println!(
            "Multiclass prediction ({} classifiers): {:?} per prediction",
            expected_classifiers,
            predict_elapsed / iterations
        );
    }

    #[test]
    #[ignore] // Run with --ignored for expensive tests
    fn bench_multiclass_large() {
        // 200 classes = 19900 classifiers
        let num_classes = 200;
        let (samples, labels) = generate_data(2000, num_classes, 20, 10000);

        let start = Instant::now();
        let mut svm = MulticlassSVM::new();
        svm.train_sparse(&samples, &labels);
        let train_elapsed = start.elapsed();

        let expected_classifiers = (num_classes * (num_classes - 1) / 2) as usize;
        println!(
            "Multiclass SVM ({} classes, {} classifiers, {} samples): {:?}",
            num_classes, expected_classifiers, samples.len(), train_elapsed
        );

        // Benchmark prediction
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = svm.predict_sparse(&samples[0]);
        }
        let predict_elapsed = start.elapsed();
        println!(
            "Multiclass prediction ({} classifiers): {:?} per prediction",
            expected_classifiers,
            predict_elapsed / iterations
        );
    }

    #[test]
    #[ignore] // Run with --ignored for very expensive tests
    fn bench_multiclass_scale_projection() {
        // Test at increasing scales to project 5000-class performance
        println!("\n=== Scale projection for 5000 classes ===");
        println!("Classes | Classifiers | Train Time | Predict Time");
        println!("--------|-------------|------------|-------------");

        for &num_classes in &[10, 25, 50, 100] {
            let samples_per_class = 10;
            let (samples, labels) = generate_data(
                num_classes as usize * samples_per_class,
                num_classes,
                20,
                10000,
            );

            let start = Instant::now();
            let mut svm = MulticlassSVM::new();
            svm.train_sparse(&samples, &labels);
            let train_elapsed = start.elapsed();

            let start = Instant::now();
            for _ in 0..100 {
                let _ = svm.predict_sparse(&samples[0]);
            }
            let predict_elapsed = start.elapsed() / 100;

            let num_classifiers = (num_classes * (num_classes - 1) / 2) as usize;
            println!(
                "{:>7} | {:>11} | {:>10.2?} | {:>10.2?}",
                num_classes, num_classifiers, train_elapsed, predict_elapsed
            );
        }

        // Extrapolate to 5000 classes
        println!("\nProjected for 5000 classes:");
        println!("  Classifiers: {}", 5000 * 4999 / 2);
        println!("  (Run with actual data for real numbers)");
    }

    #[test]
    fn bench_ovr_vs_ovo_small() {
        // Compare One-vs-Rest vs One-vs-One for 10 classes
        let num_classes = 10;
        let (samples, labels) = generate_data(200, num_classes, 20, 5000);

        println!("\n=== OvR vs OvO comparison ({} classes) ===", num_classes);

        // One-vs-One
        let start = Instant::now();
        let mut ovo = MulticlassSVM::new();
        ovo.train_sparse(&samples, &labels);
        let ovo_train = start.elapsed();

        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = ovo.predict_sparse(&samples[0]);
        }
        let ovo_predict = start.elapsed() / iterations;

        // One-vs-Rest
        let start = Instant::now();
        let mut ovr = OneVsRestSVM::new();
        ovr.train_sparse(&samples, &labels);
        let ovr_train = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ovr.predict_sparse(&samples[0]);
        }
        let ovr_predict = start.elapsed() / iterations;

        // Accuracy comparison
        let mut ovo_correct = 0;
        let mut ovr_correct = 0;
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            if ovo.predict_sparse(sample) == label {
                ovo_correct += 1;
            }
            if ovr.predict_sparse(sample) == label {
                ovr_correct += 1;
            }
        }

        println!("          | Classifiers |    Train   |  Predict  | Accuracy");
        println!("----------|-------------|------------|-----------|----------");
        println!(
            "OvO       | {:>11} | {:>10.2?} | {:>9.2?} | {:>5.1}%",
            ovo.num_classifiers(),
            ovo_train,
            ovo_predict,
            100.0 * ovo_correct as f64 / samples.len() as f64
        );
        println!(
            "OvR       | {:>11} | {:>10.2?} | {:>9.2?} | {:>5.1}%",
            ovr.num_classifiers(),
            ovr_train,
            ovr_predict,
            100.0 * ovr_correct as f64 / samples.len() as f64
        );
    }

    #[test]
    fn bench_ovr_vs_ovo_medium() {
        // Compare at 50 classes
        let num_classes = 50;
        let (samples, labels) = generate_data(500, num_classes, 20, 5000);

        println!("\n=== OvR vs OvO comparison ({} classes) ===", num_classes);

        // One-vs-One
        let start = Instant::now();
        let mut ovo = MulticlassSVM::new();
        ovo.train_sparse(&samples, &labels);
        let ovo_train = start.elapsed();

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = ovo.predict_sparse(&samples[0]);
        }
        let ovo_predict = start.elapsed() / iterations;

        // One-vs-Rest
        let start = Instant::now();
        let mut ovr = OneVsRestSVM::new();
        ovr.train_sparse(&samples, &labels);
        let ovr_train = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ovr.predict_sparse(&samples[0]);
        }
        let ovr_predict = start.elapsed() / iterations;

        // Accuracy comparison
        let mut ovo_correct = 0;
        let mut ovr_correct = 0;
        for (sample, &label) in samples.iter().zip(labels.iter()) {
            if ovo.predict_sparse(sample) == label {
                ovo_correct += 1;
            }
            if ovr.predict_sparse(sample) == label {
                ovr_correct += 1;
            }
        }

        println!("          | Classifiers |    Train   |  Predict  | Accuracy");
        println!("----------|-------------|------------|-----------|----------");
        println!(
            "OvO       | {:>11} | {:>10.2?} | {:>9.2?} | {:>5.1}%",
            ovo.num_classifiers(),
            ovo_train,
            ovo_predict,
            100.0 * ovo_correct as f64 / samples.len() as f64
        );
        println!(
            "OvR       | {:>11} | {:>10.2?} | {:>9.2?} | {:>5.1}%",
            ovr.num_classifiers(),
            ovr_train,
            ovr_predict,
            100.0 * ovr_correct as f64 / samples.len() as f64
        );
    }

    #[test]
    #[ignore]
    fn bench_ovr_scale_projection() {
        // Test OvR at increasing scales
        println!("\n=== OvR scale projection for 5000 classes ===");
        println!("Classes | Classifiers |    Train   |  Predict  | Accuracy");
        println!("--------|-------------|------------|-----------|----------");

        for &num_classes in &[10, 25, 50, 100] {
            let samples_per_class = 10;
            let (samples, labels) = generate_data(
                num_classes as usize * samples_per_class,
                num_classes,
                20,
                10000,
            );

            let start = Instant::now();
            let mut svm = OneVsRestSVM::new();
            svm.train_sparse(&samples, &labels);
            let train_elapsed = start.elapsed();

            let start = Instant::now();
            let iterations = 100;
            for _ in 0..iterations {
                let _ = svm.predict_sparse(&samples[0]);
            }
            let predict_elapsed = start.elapsed() / iterations;

            // Accuracy
            let correct = samples
                .iter()
                .zip(labels.iter())
                .filter(|(s, &l)| svm.predict_sparse(s) == l)
                .count();
            let accuracy = 100.0 * correct as f64 / samples.len() as f64;

            println!(
                "{:>7} | {:>11} | {:>10.2?} | {:>9.2?} | {:>5.1}%",
                num_classes,
                svm.num_classifiers(),
                train_elapsed,
                predict_elapsed,
                accuracy
            );
        }

        println!("\nProjected for 5000 classes with OvR:");
        println!("  Classifiers: 5000 (vs 12.5M for OvO)");
    }
}
