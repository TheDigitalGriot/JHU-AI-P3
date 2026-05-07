// All real numbers extracted from the JHU brain tumor notebook v6.0.2

export type ClassKey = 'glioma' | 'meningioma' | 'notumor' | 'pituitary'

export interface ModelRow {
  id: string
  name: string
  macroRecall: number
  worstClass: number
  f1: number
  acc: number
  trainF1: number
  gap: number
  winner?: boolean
}

export const DATA = {
  classes: ['glioma', 'meningioma', 'notumor', 'pituitary'] as ClassKey[],
  classLabels: {
    glioma: 'Glioma',
    meningioma: 'Meningioma',
    notumor: 'No Tumor',
    pituitary: 'Pituitary',
  } as Record<ClassKey, string>,
  classColors: {
    glioma: '#4dffd0',
    meningioma: '#ffb84d',
    notumor: '#9affc1',
    pituitary: '#ff6dd0',
  } as Record<ClassKey, string>,
  counts: { glioma: 645, meningioma: 660, notumor: 840, pituitary: 750 } as Record<ClassKey, number>,
  total: 2895,
  imageDims: 'tumor classes ~512×512 (tight); notumor 150–1920 (variable)',
  resizeTarget: '128×128',
  splits: { train: 2026, val: 434, test: 435 },
  meanIntensity: { glioma: 37.16, meningioma: 40.79, notumor: 61.04, pituitary: 48.07 } as Record<ClassKey, number>,
  intensityStats: {
    glioma:     { mean: 37.16, std: 8.57,  min: 13.69, q1: 31.24, median: 37.23, q3: 42.90, max: 68.36 },
    meningioma: { mean: 40.79, std: 7.85,  min: 18.23, q1: 34.80, median: 40.89, q3: 45.93, max: 62.64 },
    notumor:    { mean: 61.04, std: 21.27, min: 18.23, q1: 46.23, median: 57.16, q3: 75.24, max: 125.14 },
    pituitary:  { mean: 48.07, std: 8.15,  min: 24.70, q1: 42.19, median: 48.61, q3: 53.42, max: 76.04 },
  },
  // Audit 1 — Scanner-Fingerprint Leakage (logistic regression on per-image intensity stats only)
  leakage: {
    chance: 0.25,
    testAcc: 0.6575,
    verdict: 'PROBABLE — intensity statistics alone recover 65.8% 4-class accuracy.',
  },
  // Audit 3 — pHash slice leakage (% test images with near-duplicate in train, Hamming ≤ 4)
  phashSliceLeakage: {
    glioma: 0.165,
    meningioma: 0.182,
    notumor: 0.698,
    pituitary: 0.204,
  } as Record<ClassKey, number>,
  // Validation-set leaderboard (n = 434)
  models: [
    { id: 'ann',       name: 'Simple ANN',          macroRecall: 0.8694, worstClass: 0.7010, f1: 0.8792, acc: 0.8802, trainF1: 0.9995, gap: 0.1203 },
    { id: 'opt',       name: 'Optimized ANN',       macroRecall: 0.8809, worstClass: 0.7629, f1: 0.8891, acc: 0.8894, trainF1: 0.9951, gap: 0.1059 },
    { id: 'vgg16',     name: 'VGG-16 Base',         macroRecall: 0.8814, worstClass: 0.7732, f1: 0.8893, acc: 0.8894, trainF1: 0.9603, gap: 0.0710 },
    { id: 'vgg16ff',   name: 'VGG-16 + FF',         macroRecall: 0.8817, worstClass: 0.7320, f1: 0.8904, acc: 0.8917, trainF1: 0.9312, gap: 0.0407, winner: true },
    { id: 'vgg16aug',  name: 'VGG-16 + FF + Aug',   macroRecall: 0.8511, worstClass: 0.7216, f1: 0.8600, acc: 0.8641, trainF1: 0.8945, gap: 0.0345 },
  ] satisfies ModelRow[],
  // 5-fold CV mean ± std (macro-recall) on the train+val pool
  kfold: {
    ann:      { mean: 0.8630, std: 0.0241 },
    opt:      { mean: 0.8126, std: 0.0231 },
    vgg16:    { mean: 0.7997, std: 0.0169 },
    vgg16ff:  { mean: 0.8954, std: 0.0450 },
    vgg16aug: { mean: 0.8887, std: 0.0384 },
  } as Record<string, { mean: number; std: number }>,
  // Per-class validation recall
  perClass: {
    ann:       { glioma: 0.7010, meningioma: 0.8283, notumor: 0.9841, pituitary: 0.9643 },
    opt:       { glioma: 0.7629, meningioma: 0.8283, notumor: 0.9683, pituitary: 0.9643 },
    vgg16:     { glioma: 0.7732, meningioma: 0.8586, notumor: 0.9921, pituitary: 0.9018 },
    vgg16ff:   { glioma: 0.7320, meningioma: 0.8485, notumor: 1.0000, pituitary: 0.9464 },
    vgg16aug:  { glioma: 0.7216, meningioma: 0.7273, notumor: 1.0000, pituitary: 0.9554 },
  } as Record<string, Record<ClassKey, number>>,
  // Test-set confusion matrix for the rubric winner — VGG-16 + FF (n = 435)
  // Diagonals: glioma 66/97 (0.6804), meningioma 78/99 (0.7879), notumor 125/126 (0.9921), pituitary 104/113 (0.9204)
  // Off-diagonals reconstructed from per-class precision in the classification report (glioma↔meningioma dominant).
  confusion: {
    vgg16ff: [
      [66, 28,  0,  3],
      [13, 78,  0,  8],
      [ 0,  1, 125,  0],
      [ 4,  5,  0, 104],
    ],
  } as Record<string, number[][]>,
  // Final test-set headline (rubric VGG-16+FF) and deployment ensembles
  finalTest: {
    rubricWinner: {
      name: 'VGG-16 + FF',
      macroRecall: 0.8452,
      ciLow: 0.8113, ciHigh: 0.8783,
      perClass: { glioma: 0.6804, meningioma: 0.7879, notumor: 0.9921, pituitary: 0.9204 },
      f1: 0.8574, acc: 0.8575,
      worstClassFloorCleared: false,
    },
    keras5FoldOOFStacker: {
      macroRecall: 0.9165,
      ciLow: 0.8905, ciHigh: 0.9421,
      perClass: { glioma: 0.8351, meningioma: 0.8485, notumor: 1.0000, pituitary: 0.9823 },
      worstClassFloorCleared: false,
    },
    pytorchEnsemble: {
      macroRecall: 0.9700,
      ciLow: 0.9525, ciHigh: 0.9855,
      perClass: { glioma: 0.9381, meningioma: 0.9596, notumor: 1.0000, pituitary: 0.9823 },
      worstClassFloorCleared: true,
    },
  },
  // Calibration — Expected Calibration Error (ECE)
  calibration: {
    kerasWinner: { rawECE: 0.0436, scaledECE: 0.0448, T: 0.955, deltaPct: -0.029 },
    pytorchL2Stacker: {
      rawECE: 0.0186,
      perClassScaledECE: 0.0144,
      deltaPct: 0.226,
      perClassT: { glioma: 0.9186, meningioma: 0.9529, notumor: 0.6303, pituitary: 0.9350 },
    },
  },
  // View-orientation audit — Part 2 stratified test eval (median symmetry split)
  viewOrientation: {
    high: { glioma: 0.6034, meningioma: 0.7447, notumor: 1.0000, pituitary: 0.9138, macro: 0.8155 },
    low:  { glioma: 0.7949, meningioma: 0.8269, notumor: 0.9859, pituitary: 0.9273, macro: 0.8837 },
    delta: { glioma: -0.1914, meningioma: -0.0822, notumor: 0.0141, pituitary: -0.0135, macro: -0.0683 },
    confirmed: 'meningioma → sagittal shortcut (high-sym 0.7447 vs low-sym 0.8269 raw; +8.2 pp easier when sagittal-like)',
    rejected: 'notumor → axial shortcut',
  },
  // Reject-option frontier (Part 4) — confidence threshold τ vs coverage
  rejectOption: [
    { tau: 0.50, coverage: 0.9724, macroR: 0.8532, macroP: 0.8589 },
    { tau: 0.60, coverage: 0.8920, macroR: 0.8671, macroP: 0.8731 },
    { tau: 0.70, coverage: 0.8161, macroR: 0.9075, macroP: 0.9153 },
    { tau: 0.80, coverage: 0.7356, macroR: 0.9362, macroP: 0.9379 },
    { tau: 0.85, coverage: 0.6644, macroR: 0.9586, macroP: 0.9617 },
    { tau: 0.90, coverage: 0.6046, macroR: 0.9557, macroP: 0.9535 },
    { tau: 0.95, coverage: 0.5011, macroR: 0.9463, macroP: 0.9581 },
  ],
  // Deep-ensemble mutual information (Part 6)
  deepEnsembleMI: {
    glioma:     { meanMI: 0.1154, medianMI: 0.0422, q75: 0.2074 },
    meningioma: { meanMI: 0.1385, medianMI: 0.0935, q75: 0.2437 },
    notumor:    { meanMI: 0.0023, medianMI: 0.0000, q75: 0.0000 },
    pituitary:  { meanMI: 0.0288, medianMI: 0.0003, q75: 0.0070 },
    lowMIRecall: 1.0000,
    highMIRecall: 0.8737,
  },
  // Clinical / methodological thresholds
  thresholds: {
    macroRecallTarget: 0.866,
    worstClassFloor: 0.866,
    gliomaTarget: 0.866,
  },
  // WHO-grade-informed Bayes-risk loss matrix L[true, pred]
  // Cost of missing glioma = 4, cost of false notumor = 3, off-diagonals otherwise = 1.
  costMatrix: [
    [0, 4, 4, 4],   // glioma row    — every miss is treatment-critical
    [1, 0, 3, 1],   // meningioma row — false notumor weighted 3x
    [1, 1, 0, 1],   // notumor row    — uniform off-diagonal cost
    [1, 1, 3, 0],   // pituitary row  — false notumor weighted 3x
  ],
} as const

export const SNIPPETS = {
  imports: `from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split, StratifiedKFold`,

  splits: `# 70 / 15 / 15 stratified split — class balance preserved
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.30, stratify=labels, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Training set   : 2026 images (70.0%)
# Validation set : 434 images  (15.0%)
# Test set       : 435 images  (15.0%)`,

  normalize: `# Min–max scale to [0, 1] for numerical stability
X_train_n = X_train.astype('float32') / 255.0
X_val_n   = X_val.astype('float32')   / 255.0
X_test_n  = X_test.astype('float32')  / 255.0`,

  leakage: `# Leakage Audit Triad — three independent diagnostics
# 1. Scanner fingerprint: can intensity stats alone predict the class?
features = [mean, std, median, skew, kurtosis]   # per-image, per-channel
clf = LogisticRegression(class_weight='balanced').fit(X_train_stats, y_train)
print(clf.score(X_test_stats, y_test))
# 0.6575  vs  0.25 chance  →  PROBABLE leakage`,

  ann: `# Simple ANN — flatten 128×128×3 = 49,152 features into a vector
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])
# All spatial information is destroyed at the Flatten step.`,

  opt: `# Optimized ANN — BatchNorm + Dropout + LR schedule
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(512), BatchNormalization(), ReLU(), Dropout(0.4),
    Dense(256), BatchNormalization(), ReLU(), Dropout(0.3),
    Dense(4, activation='softmax')
])
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')`,

  vgg16: `# VGG-16 base — frozen ImageNet feature extractor
base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
for layer in base.layers: layer.trainable = False

model = Sequential([
    Lambda(vgg16_preprocess),     # [0,1] RGB → BGR ImageNet
    base,                         # 14,714,688 frozen params
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')  # only 2,052 trainable
])`,

  vgg16ff: `# VGG-16 + Feedforward head (rubric Final Model)
model = Sequential([
    Lambda(vgg16_preprocess),
    vgg16_base,                   # frozen
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])`,

  vgg16aug: `# MRI-safe augmentation — NO rotation, NO shear (Pérez-García 2021)
aug = ImageDataGenerator(
    horizontal_flip=True,    # safe for axial MRI
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    fill_mode='nearest',
)`,

  kfold: `# 5-fold StratifiedKFold variance estimation on the train+val pool
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(skf.split(X_pool, y_pool.argmax(1))):
    model = build_fresh()                          # fresh init each fold
    model.fit(X_pool[tr], y_pool[tr],
              validation_data=(X_pool[va], y_pool[va]),
              epochs=25, callbacks=[EarlyStopping('val_loss', patience=10)])
    fold_scores.append(macro_recall(y_pool[va], model.predict(X_pool[va])))
# pHash catches data leakage; K-fold catches split luck.`,

  ensemble: `# True 5-Fold OOF Keras Stacker  +  PyTorch 4-Base Ensemble
# Both stackers use Wolpert (1992) discipline: meta-LR fit on out-of-fold predictions.
keras_oof = np.concatenate([model.predict(X_pool[va]) for _, va in skf.split(...)])
meta_lr   = LogisticRegression(C=1.0).fit(keras_oof, y_pool)

# PyTorch ensemble: 4 timm bases × 5 folds, BF16-AMP, per-class temperature scaling
pytorch_test_macro_recall = 0.9700   # [95% CI  0.9525, 0.9855]
keras_test_macro_recall   = 0.9165   # [95% CI  0.8905, 0.9421]
# Single rubric VGG-16+FF  : 0.8452   [glioma 0.6804 — below 0.866 floor]`,
} as const
