# Experiment 2: Multilayer Perceptron (MLP) for XOR Gate & Hyperparameter Tuning
# Student Name: Rahul Bharathwaj
# Student Roll Number: CH.SC.U4AIE24058

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=" * 60)
print("Student Name: Rahul Bharathwaj")
print("Student Roll Number: CH.SC.U4AIE24058")
print("Experiment 2: MLP for XOR Classification & Hyperparameter Tuning")
print("=" * 60)

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Configurations to tune
configs = [
    {"activation": "relu", "lr": 0.01, "desc": "ReLU + LR 0.01"},
    {"activation": "relu", "lr": 0.1,  "desc": "ReLU + LR 0.10"},
    {"activation": "sigmoid", "lr": 0.01, "desc": "Sigmoid + LR 0.01"},
    {"activation": "sigmoid", "lr": 0.1,  "desc": "Sigmoid + LR 0.10"}
]

results = []

for cfg in configs:
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(8, activation=cfg["activation"]),
        layers.Dense(4, activation=cfg["activation"]),
        layers.Dense(1, activation="sigmoid")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X, y, epochs=50, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    preds = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()

    results.append({"desc": cfg["desc"], "loss": loss, "acc": acc, "preds": preds})
    print(f"[Roll No: CH.SC.U4AIE24058] {cfg['desc']} -> Loss: {loss:.4f}, Accuracy: {acc*100:.1f}%, Predictions: {preds}")

print("\n" + "=" * 60)
print(f"Hyperparameter Comparison Summary [Roll No: CH.SC.U4AIE24058]:")
print("=" * 60)
for r in results:
    print(f"Config: {r['desc']:<20} | Final Loss: {r['loss']:.4f} | Accuracy: {r['acc']*100:.1f}% | Output: {r['preds']}")
