# Experiment 9: Transfer Learning with MobileNetV2 on Local SVHN
# Student Name: Rahul Bharathwaj
# Student Roll Number: CH.SC.U4AIE24058

from pathlib import Path

import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("=" * 60)
print("Student Name: Rahul Bharathwaj")
print("Student Roll Number: CH.SC.U4AIE24058")
print("Experiment 9: MobileNetV2 Transfer Learning on Local SVHN")
print("=" * 60)

# Load the workspace-local SVHN files; do not request remote datasets.
data_dir = Path(__file__).resolve().parent / "data"
train_data = loadmat(data_dir / "train_32x32.mat")
test_data = loadmat(data_dir / "test_32x32.mat")
X_train_full = np.transpose(train_data["X"], (3, 0, 1, 2))
y_train_full = train_data["y"].reshape(-1).astype("int32") % 10
X_test_full = np.transpose(test_data["X"], (3, 0, 1, 2))
y_test_full = test_data["y"].reshape(-1).astype("int32") % 10

# Use a deterministic subset for a short, reproducible CPU lab run.
rng = np.random.default_rng(42)
train_indices = rng.choice(len(X_train_full), size=1000, replace=False)
test_indices = rng.choice(len(X_test_full), size=300, replace=False)
X_train = X_train_full[train_indices]
y_train = y_train_full[train_indices]
X_test = X_test_full[test_indices]
y_test = y_test_full[test_indices]
print(f"Loaded local SVHN: {len(X_train_full)} train and {len(X_test_full)} test images.")
print(f"Training on {len(X_train)} images; evaluating on {len(X_test)} images.")

# Build Preprocessing & Resizing Model
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Resizing(96, 96)(inputs)
x = layers.Lambda(preprocess_input)(x)

# Load MobileNetV2 Base
weights_path = Path.home() / ".keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
if not weights_path.is_file():
    raise FileNotFoundError(f"Cached MobileNetV2 ImageNet weights not found: {weights_path}")
base_model = MobileNetV2(weights=str(weights_path), include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# Phase 1: Train Top Classifier Layers
print(f"\n[Roll No: CH.SC.U4AIE24058] Phase 1: Training Classification Head (Base Frozen)...")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_phase1 = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test), verbose=2)

loss_p1, acc_p1 = model.evaluate(X_test, y_test, verbose=0)
print(f"[Roll No: CH.SC.U4AIE24058] Phase 1 Test Accuracy: {acc_p1 * 100:.2f}%")

# Phase 2: Fine-Tuning Last 20 Layers
print(f"\n[Roll No: CH.SC.U4AIE24058] Phase 2: Unfreezing Last 20 Layers for Fine-Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history_phase2 = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=2)

loss_p2, acc_p2 = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 60)
print(f"Transfer Learning Results [Roll No: CH.SC.U4AIE24058]:")
print(f"Frozen-base SVHN test accuracy:    {acc_p1 * 100:.2f}%")
print(f"Fine-tuned SVHN test accuracy:     {acc_p2 * 100:.2f}%")
print("=" * 60)
