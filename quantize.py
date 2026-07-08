"""
quantize.py
-----------
Post-Training Quantization (PTQ) for the BISINDO sign language model.

Converts a trained Keras/SavedModel into a TFLite flatbuffer using one of
three quantization strategies:

    1. float16   - Weights quantized to FP16.  Fastest to produce, minimal
                   accuracy loss, ~2x size reduction. GPU-friendly.
    2. dynamic   - Weights quantized to INT8 at conversion time; activations
                   quantized dynamically at inference. No calibration data
                   needed.  Good CPU speed-up, ~4x size reduction.
    3. int8      - Full-integer quantization (weights + activations). Requires
                   a representative calibration dataset built from the real
                   .npy feature files.  Best latency on MCUs/Edge TPUs.

Usage
-----
    # Default: dynamic-range quantization
    python quantize.py

    # Choose a specific strategy
    python quantize.py --mode float16
    python quantize.py --mode dynamic
    python quantize.py --mode int8

    # Specify a custom model path or output path
    python quantize.py --model signlingo_v2_gru.h5 --output tflite_models/

    # Evaluate model accuracy after quantization
    python quantize.py --mode int8 --evaluate
"""

import argparse
import os
import time
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuration (keep in sync with train.py)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30
FEATURE_DIM     = 447   # matches train.py FEATURES_DIM
DATA_PATH       = 'features'
MODEL_SAVE_PATH = 'signlingo_v2_gru.h5'

def get_actions():
    """Load action labels dynamically from the features directory (same as train.py)."""
    if not os.path.exists(DATA_PATH):
        return []
    return sorted([d for d in os.listdir(DATA_PATH)
                   if os.path.isdir(os.path.join(DATA_PATH, d))])


# ---------------------------------------------------------------------------
# Representative dataset generator (used for INT8 calibration)
# ---------------------------------------------------------------------------
def make_representative_dataset(num_samples: int = 200):
    """
    Yields batches of shape (1, SEQUENCE_LENGTH, FEATURE_DIM) sampled
    randomly from the extracted .npy feature files.
    """
    all_sequences = []
    actions = get_actions()

    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        for fname in os.listdir(action_dir):
            if not fname.endswith('.npy'):
                continue
            fpath = os.path.join(action_dir, fname)
            try:
                seq = np.load(fpath)  # shape: (SEQUENCE_LENGTH, FEATURE_DIM)
                if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                    all_sequences.append(seq.astype(np.float32))
            except Exception as e:
                print(f"  [WARN] Could not load {fpath}: {e}")

    if not all_sequences:
        raise RuntimeError(
            f"No valid .npy files found under '{DATA_PATH}/'. "
            "Run extraction.py first to generate feature files."
        )

    # Shuffle and cap at num_samples
    np.random.shuffle(all_sequences)
    all_sequences = all_sequences[:num_samples]
    print(f"  Calibration dataset: {len(all_sequences)} sequences "
          f"(shape per sample: {all_sequences[0].shape})")

    def generator():
        for seq in all_sequences:
            yield [seq[np.newaxis, ...]]  # Add batch dimension -> (1, 30, FEATURE_DIM)

    return generator


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------
def quantize(
    model_path: str,
    output_dir: str,
    mode: str,
    num_calibration_samples: int = 200,
) -> str:
    """
    Loads a Keras model and converts it to TFLite using the chosen strategy.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (.keras / .h5 / SavedModel directory).
    output_dir : str
        Directory where the .tflite file will be written.
    mode : str
        One of 'float16', 'dynamic', 'int8'.
    num_calibration_samples : int
        Number of feature sequences to use for INT8 calibration.

    Returns
    -------
    str
        Absolute path to the generated .tflite file.
    """
    print(f"\n{'='*60}")
    print(f"  BISINDO Sign Language Model Quantization")
    print(f"  Mode   : {mode.upper()}")
    print(f"  Source : {model_path}")
    print(f"{'='*60}\n")

    # --- Load model ---------------------------------------------------------
    print("[1/3] Loading model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Train and save your model first (e.g. model.save('bisindo_model.keras'))."
        )
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # --- Configure converter ------------------------------------------------
    print(f"\n[2/3] Converting to TFLite ({mode} quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        suffix = 'float16'

    elif mode == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        suffix = 'dynamic'

    elif mode == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(
            num_calibration_samples
        )
        # Force full-integer ops; fall back to float for unsupported ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type  = tf.float32  # keep float I/O for ease-of-use
        converter.inference_output_type = tf.float32
        suffix = 'int8'

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: float16, dynamic, int8")

    t0 = time.time()
    tflite_model = converter.convert()
    elapsed = time.time() - t0
    print(f"  Conversion completed in {elapsed:.1f}s")

    # --- Save .tflite file --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"bisindo_{suffix}.tflite"
    out_path = os.path.join(output_dir, out_filename)

    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    original_size = os.path.getsize(model_path) if os.path.isfile(model_path) else None
    tflite_size   = os.path.getsize(out_path)

    print(f"\n[3/3] Saved -> {out_path}")
    print(f"  TFLite model size : {tflite_size / 1024:.1f} KB")
    if original_size:
        reduction = (1 - tflite_size / original_size) * 100
        print(f"  Original size     : {original_size / 1024:.1f} KB  ({original_size / 1024 / 1024:.2f} MB)")
        print(f"  Size reduction    : {reduction:.1f}%")

    return out_path


# ---------------------------------------------------------------------------
# Optional: evaluate quantized model accuracy
# ---------------------------------------------------------------------------
def evaluate_tflite(tflite_path: str):
    """
    Runs inference on all feature sequences and prints top-1 accuracy.
    Actions are loaded dynamically from the features/ folder (same as train.py).
    """
    print(f"\n{'='*60}")
    print("  Evaluating quantized model accuracy...")
    print(f"{'='*60}\n")

    actions = get_actions()
    if not actions:
        print(f"  No action folders found under '{DATA_PATH}/'.")
        return

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total   = 0

    for label_idx, action in enumerate(actions):
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_dir):
            continue
        for fname in os.listdir(action_dir):
            if not fname.endswith('.npy'):
                continue
            seq = np.load(os.path.join(action_dir, fname))
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                continue

            input_data = seq[np.newaxis, ...].astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            predicted = np.argmax(output[0])
            if predicted == label_idx:
                correct += 1
            total += 1

    if total == 0:
        print("  No sequences found for evaluation. Run extraction.py first.")
        return

    accuracy = correct / total * 100
    print(f"  Samples evaluated : {total}")
    print(f"  Correct           : {correct}")
    print(f"  Top-1 Accuracy    : {accuracy:.2f}%")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Training Quantization for BISINDO sign language model."
    )
    parser.add_argument(
        '--model', type=str, default=MODEL_SAVE_PATH,
        help=f"Path to the trained model file (default: {MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        '--output', type=str, default='tflite_models',
        help="Output directory for .tflite files (default: tflite_models/)"
    )
    parser.add_argument(
        '--mode', type=str, default='dynamic',
        choices=['float16', 'dynamic', 'int8'],
        help=(
            "Quantization strategy: "
            "float16 (FP16 weights, ~2x smaller), "
            "dynamic (INT8 weights, ~4x smaller), "
            "int8 (full INT8, needs feature data). "
            "Default: dynamic"
        )
    )
    parser.add_argument(
        '--calibration-samples', type=int, default=200,
        help="Number of .npy sequences for INT8 calibration (default: 200)"
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help="Evaluate quantized model accuracy after conversion"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    tflite_path = quantize(
        model_path=args.model,
        output_dir=args.output,
        mode=args.mode,
        num_calibration_samples=args.calibration_samples,
    )

    if args.evaluate:
        evaluate_tflite(tflite_path)

    print(f"\nDone! TFLite model saved to: {tflite_path}")
