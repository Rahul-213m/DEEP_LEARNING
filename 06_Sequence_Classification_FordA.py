# Experiment 6: Sequence Classification on Local FordA Data
# Student Name: Rahul Bharathwaj
# Student Roll Number: CH.SC.U4AIE24058
# Uses only FordA files present in this workspace.
# Stock-price and IMDB tasks are skipped because their datasets are absent.

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, SimpleRNN
from sklearn.metrics import classification_report

print("Student Name: Rahul Bharathwaj | Roll No: CH.SC.U4AIE24058")
print(f"TensorFlow Version: {tf.__version__}")

dataset_dir = Path(__file__).resolve().parent / "raw.githubusercontent.com" / "hfawaz" / "cd-diagram" / "master" / "FordA"
train_path = dataset_dir / "FordA_TRAIN.tsv"
test_path = dataset_dir / "FordA_TEST.tsv"

print("\nStock-price prediction skipped: no stock-price dataset is available in this workspace.")
print("IMDB SimpleRNN skipped: no IMDB review dataset is available in this workspace.")
print("IMDB LSTM skipped: no IMDB review dataset is available in this workspace.")

if not train_path.is_file() or not test_path.is_file():
    print(f"FordA recurrent classification skipped: local dataset files not found in {dataset_dir}.")
else:
    train_data = np.loadtxt(train_path, delimiter="\t")
    test_data = np.loadtxt(test_path, delimiter="\t")
    X_train = train_data[:, 1:].astype("float32")
    X_test = test_data[:, 1:].astype("float32")
    y_train = (train_data[:, 0] == 1).astype("int32")
    y_test = (test_data[:, 0] == 1).astype("int32")

    mean = X_train.mean()
    std = X_train.std()
    X_train = ((X_train - mean) / std)[..., np.newaxis]
    X_test = ((X_test - mean) / std)[..., np.newaxis]

    rng = np.random.default_rng(42)
    sample_size = min(1200, len(X_train))
    train_indices = rng.choice(len(X_train), size=sample_size, replace=False)
    X_fit = X_train[train_indices]
    y_fit = y_train[train_indices]
    print(f"\nLoaded local FordA: {len(X_train)} training and {len(X_test)} test sequences.")
    print(f"Training on a reproducible subset of {len(X_fit)} sequences; testing on all {len(X_test)} test sequences.")
    print(f"Sequence shape: {X_train.shape[1:]} | Classes: 0 and 1")

    tf.keras.utils.set_random_seed(42)

    def train_and_evaluate(model_name, recurrent_layer):
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            recurrent_layer(32),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print(f"\nTraining {model_name} on FordA...")
        model.fit(X_fit, y_fit, epochs=3, batch_size=64, verbose=2)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        predictions = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype("int32")
        print(f"{model_name} FordA test loss: {loss:.4f}")
        print(f"{model_name} FordA test accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"], digits=3))

    train_and_evaluate("SimpleRNN", SimpleRNN)
    train_and_evaluate("LSTM", LSTM)
