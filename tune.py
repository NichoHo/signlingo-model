"""
tune.py
-------
Hyperparameter tuning for the BISINDO sign language GRU classifier
using Keras Tuner (Hyperband search strategy).

Mirrors the architecture and data pipeline in train.py exactly:
    - GRU-based Sequential model
    - FEATURES_DIM = 447, SEQUENCE_LENGTH = 30, NUM_CLASSES = 40
    - One-hot encoded labels (categorical_crossentropy)
    - 80 / 10 / 10 train / val / test split

Searches over:
    - GRU units (layer 1 & 2)
    - Dropout rate
    - Learning rate

After tuning, the best hyperparameters are printed and the best model is
retrained and saved as  signlingo_v2_gru.h5  (same output as train.py).

Usage
-----
    pip install keras-tuner

    python tune.py
    python tune.py --max-trials 20 --tune-epochs 50
    python tune.py --retrain-epochs 150 --output my_best_model.h5
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

try:
    import keras_tuner as kt
except ImportError:
    raise ImportError(
        "keras-tuner is not installed.\n"
        "Install it with:  pip install keras-tuner"
    )

# ---------------------------------------------------------------------------
# Configuration  (kept in sync with train.py)
# ---------------------------------------------------------------------------
DATA_DIR        = "features"
SEQUENCE_LENGTH = 30
FEATURES_DIM    = 447
NUM_CLASSES     = 40
MODEL_SAVE_PATH = "signlingo_v2_gru.h5"


# ---------------------------------------------------------------------------
# Data loading  (identical to train.py's load_data)
# ---------------------------------------------------------------------------
def load_data(data_dir: str):
    """
    Dynamically iterates through the features directory to load .npy files.
    Returns X, y (one-hot), actions list, and num_classes — same as train.py.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Run extraction.py first.")
        return np.array([]), np.array([]), [], 0

    actions = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if len(actions) != NUM_CLASSES:
        print(f"Warning: Expected {NUM_CLASSES} classes, found {len(actions)}.")

    num_classes = len(actions)
    label_map   = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(data_dir, action)
        for sequence_file in os.listdir(action_path):
            if sequence_file.endswith('.npy'):
                res = np.load(os.path.join(action_path, sequence_file))
                if res.shape == (SEQUENCE_LENGTH, FEATURES_DIM):
                    sequences.append(res)
                    labels.append(label_map[action])
                else:
                    print(f"Warning: Skipping {sequence_file} in {action} "
                          f"due to incorrect shape {res.shape}")

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=num_classes).astype(int)
    return X, y, actions, num_classes


# ---------------------------------------------------------------------------
# Hypermodel (callable by Keras Tuner)
# ---------------------------------------------------------------------------
def build_hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
    """
    GRU search space that mirrors train.py's architecture.

    Search space
    ------------
    gru_units_1   : {32, 64, 96, 128}
    gru_units_2   : {32, 64, 96, 128}
    dropout_rate  : float in [0.1, 0.5], step 0.05
    l2_lambda     : log-uniform in [1e-4, 1e-2]
    learning_rate : log-uniform in [1e-4, 1e-2]
    """
    gru_units_1   = hp.Choice('gru_units_1',   [32, 64, 96, 128])
    gru_units_2   = hp.Choice('gru_units_2',   [32, 64, 96, 128])
    dropout_rate  = hp.Float('dropout_rate',   min_value=0.1, max_value=0.5, step=0.05)
    l2_lambda     = hp.Float('l2_lambda',      min_value=1e-4, max_value=1e-2,
                              sampling='log')
    learning_rate = hp.Float('learning_rate',  min_value=1e-4, max_value=1e-2,
                              sampling='log')

    model = Sequential([
        GRU(gru_units_1, return_sequences=True,
            input_shape=(SEQUENCE_LENGTH, FEATURES_DIM),
            kernel_regularizer=l2(l2_lambda),
            recurrent_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        GRU(gru_units_2, return_sequences=False,
            kernel_regularizer=l2(l2_lambda),
            recurrent_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(NUM_CLASSES, activation='softmax',
              kernel_regularizer=l2(l2_lambda)),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# Print & save best hyperparameters
# ---------------------------------------------------------------------------
def report_best_hparams(best_hp: kt.HyperParameters,
                         out_file: str = 'best_hparams.txt'):
    lines = [
        "=" * 50,
        "  BEST HYPERPARAMETERS",
        "=" * 50,
        f"  gru_units_1   : {best_hp.get('gru_units_1')}",
        f"  gru_units_2   : {best_hp.get('gru_units_2')}",
        f"  dropout_rate  : {best_hp.get('dropout_rate'):.2f}",
        f"  l2_lambda     : {best_hp.get('l2_lambda'):.6f}",
        f"  learning_rate : {best_hp.get('learning_rate'):.6f}",
        "=" * 50,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_file, 'w') as f:
        f.write(report + "\n")
    print(f"  Saved to: {out_file}")


# ---------------------------------------------------------------------------
# Main tuning + retrain routine
# ---------------------------------------------------------------------------
def run_tuning(
    output_path:    str   = MODEL_SAVE_PATH,
    tuner_dir:      str   = 'tuner_results',
    max_trials:     int   = 30,
    tune_epochs:    int   = 40,
    retrain_epochs: int   = 150,
    random_seed:    int   = 42,
):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # ---- Load data (same split logic as train.py) --------------------------
    print("Loading data...")
    X, y, actions, num_classes = load_data(DATA_DIR)

    if len(X) == 0:
        print("Error: No data found. Please check your data directory.")
        return

    print(f"Total samples loaded: {X.shape[0]}")
    print(f"Features shape: {X.shape}")

    # 80% train, 10% val, 10% test  (mirrors train.py exactly)
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.20, stratify=np.argmax(y, axis=1), random_state=42
        )
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50,
            stratify=np.argmax(y_temp, axis=1), random_state=42
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

    print(f"Training data   : {X_train.shape[0]} samples")
    print(f"Validation data : {X_val.shape[0]} samples")
    print(f"Testing data    : {X_test.shape[0]} samples")

    # ---- Tuner setup -------------------------------------------------------
    tuner = kt.Hyperband(
        hypermodel           = build_hypermodel,
        objective            = kt.Objective('val_accuracy', direction='max'),
        max_epochs           = tune_epochs,
        factor               = 3,
        hyperband_iterations = 2,
        directory            = tuner_dir,
        project_name         = 'bisindo_gru',
        overwrite            = False,   # set True to restart from scratch
        seed                 = random_seed,
    )

    tuner.search_space_summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=8,
                               restore_best_weights=True)

    print(f"\nStarting Hyperband search (up to {tune_epochs} epochs/trial)...\n")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    # ---- Best hyperparameters ----------------------------------------------
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    report_best_hparams(best_hp)

    # ---- Retrain best model for full epochs --------------------------------
    print(f"\nRetraining best model for up to {retrain_epochs} epochs...\n")
    best_model = build_hypermodel(best_hp)
    best_model.summary()

    cb_list = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(output_path, monitor='val_loss',
                        save_best_only=True, verbose=1),
    ]

    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=retrain_epochs,
        callbacks=cb_list,
        verbose=1,
    )

    # ---- Test evaluation ---------------------------------------------------
    print("\nEvaluating on test set...")
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Loss     : {test_loss:.4f}")
    print(f"Final Test Accuracy : {test_acc * 100:.2f}%")

    # ModelCheckpoint already saved the best weights to output_path
    print(f"\nBest model saved to: {output_path}")
    return best_model, best_hp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for BISINDO GRU model (Keras Tuner Hyperband)"
    )
    parser.add_argument('--output',         type=str, default=MODEL_SAVE_PATH,
                        help=f"Output model path (default: {MODEL_SAVE_PATH})")
    parser.add_argument('--tuner-dir',      type=str, default='tuner_results',
                        help="Directory to store tuner state (default: tuner_results/)")
    parser.add_argument('--max-trials',     type=int, default=30,
                        help="Max Hyperband trials (default: 30)")
    parser.add_argument('--tune-epochs',    type=int, default=40,
                        help="Max epochs per trial during search (default: 40)")
    parser.add_argument('--retrain-epochs', type=int, default=150,
                        help="Epochs for final retrain of best model (default: 150)")
    parser.add_argument('--seed',           type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tuning(
        output_path    = args.output,
        tuner_dir      = args.tuner_dir,
        max_trials     = args.max_trials,
        tune_epochs    = args.tune_epochs,
        retrain_epochs = args.retrain_epochs,
        random_seed    = args.seed,
    )
