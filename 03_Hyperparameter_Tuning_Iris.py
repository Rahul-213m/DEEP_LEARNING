# Experiment 3: Hyperparameter Tuning on Iris Dataset
# Student Name: Rahul Bharathwaj
# Student Roll Number: CH.SC.U4AIE24058

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("Student Name: Rahul Bharathwaj")
print("Student Roll Number: CH.SC.U4AIE24058")
print("Experiment 3: Hyperparameter Tuning for MLP on Iris Dataset")
print("=" * 60)

# Load and preprocess dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter search space
param_grid = [
    {"units": 16, "dropout": 0.1, "lr": 0.01},
    {"units": 16, "dropout": 0.2, "lr": 0.001},
    {"units": 32, "dropout": 0.1, "lr": 0.01},
    {"units": 32, "dropout": 0.2, "lr": 0.001},
]

records = []
for p in param_grid:
    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(p["units"], activation="relu"),
        layers.Dropout(p["dropout"]),
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16, verbose=0)
    train_loss = history.history["loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    cfg_str = f"Units: {p['units']}, Dropout: {p['dropout']}, LR: {p['lr']}"
    records.append((cfg_str, train_loss, val_acc))
    print(f"[Roll No: CH.SC.U4AIE24058] {cfg_str} -> Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

print("\n" + "=" * 60)
print(f"Hyperparameter Grid Results Summary [Roll No: CH.SC.U4AIE24058]:")
print("=" * 60)
for cfg_str, loss, acc in sorted(records, key=lambda x: x[2], reverse=True):
    print(f"{cfg_str:<40} | Loss: {loss:.4f} | Val Acc: {acc*100:.2f}%")
