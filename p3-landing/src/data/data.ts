// All real numbers extracted from the JHU brain tumor notebook v4.3.0

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
  leakage: {
    chance: 0.25,
    testAcc: 0.6575,
    verdict: 'PROBABLE — intensity statistics alone recover 65.8% 4-class accuracy.',
  },
  models: [
    { id: 'ann',       name: 'Simple ANN',          macroRecall: 0.8727, worstClass: 0.7320, f1: 0.8833, acc: 0.8825, trainF1: 0.9827, gap: 0.0994 },
    { id: 'opt',       name: 'Optimized ANN',       macroRecall: 0.8601, worstClass: 0.6869, f1: 0.8696, acc: 0.8687, trainF1: 0.9192, gap: 0.0495 },
    { id: 'vgg16',     name: 'VGG-16 Base',         macroRecall: 0.8938, worstClass: 0.8182, f1: 0.9013, acc: 0.9009, trainF1: 0.9525, gap: 0.0512 },
    { id: 'vgg16ff',   name: 'VGG-16 + FF',         macroRecall: 0.9233, worstClass: 0.8283, f1: 0.9281, acc: 0.9286, trainF1: 0.9762, gap: 0.0481, winner: true },
    { id: 'vgg16aug',  name: 'VGG-16 + FF + Aug',   macroRecall: 0.8734, worstClass: 0.7010, f1: 0.8829, acc: 0.8848, trainF1: 0.9081, gap: 0.0252 },
  ] satisfies ModelRow[],
  perClass: {
    ann:       { glioma: 0.7320, meningioma: 0.8384, notumor: 0.9921, pituitary: 0.9286 },
    opt:       { glioma: 0.8557, meningioma: 0.6869, notumor: 0.9603, pituitary: 0.9375 },
    vgg16:     { glioma: 0.8454, meningioma: 0.8182, notumor: 0.9921, pituitary: 0.9196 },
    vgg16ff:   { glioma: 0.9175, meningioma: 0.8283, notumor: 0.9921, pituitary: 0.9554 },
    vgg16aug:  { glioma: 0.7010, meningioma: 0.8283, notumor: 1.0000, pituitary: 0.9643 },
  } as Record<string, Record<ClassKey, number>>,
  confusion: {
    vgg16ff: [
      [89,  6, 1, 1],
      [12, 82, 2, 3],
      [ 0,  1, 125, 0],
      [ 2,  2, 1, 107],
    ],
  } as Record<string, number[][]>,
  thresholds: {
    macroRecallTarget: 0.930,
    worstClassFloor: 0.866,
    gliomaTarget: 0.88,
  },
  costMatrix: [
    [0, 3, 8, 4],
    [3, 0, 6, 4],
    [4, 4, 0, 4],
    [4, 3, 6, 0],
  ],
} as const

export const SNIPPETS = {
  imports: `from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split`,

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

  leakage: `# Scanner-Fingerprint Leakage Audit
# Can a logistic regression predict class from intensity statistics alone?
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

  vgg16ff: `# VGG-16 + Feedforward head (rubric winner)
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
)
# v4.0 fix: lr 5e-4 → 1e-4 ; patience 10 → 20 (aug adds gradient variance)`,
} as const
