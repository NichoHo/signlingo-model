import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_DIR = "features"
SEQUENCE_LENGTH = 30
FEATURES_DIM = 447
NUM_CLASSES = 40
MODEL_SAVE_PATH = "signlingo_v2_gru.h5"

def load_data(data_dir):
    """
    Dynamically iterate through the features directory to load .npy files
    and their string labels.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Please run data extraction first.")
        return np.array([]), np.array([]), [], 0
        
    actions = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(actions) != NUM_CLASSES:
        print(f"Warning: Expected {NUM_CLASSES} classes, found {len(actions)}.")
    
    num_classes = len(actions)
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    
    for action in actions:
        action_path = os.path.join(data_dir, action)
        for sequence_file in os.listdir(action_path):
            if sequence_file.endswith('.npy'):
                res = np.load(os.path.join(action_path, sequence_file))
                
                # Ensure the exact shape requirement (30, 447)
                if res.shape == (SEQUENCE_LENGTH, FEATURES_DIM):
                    sequences.append(res)
                    labels.append(label_map[action])
                else:
                    print(f"Warning: Skipping {sequence_file} in {action} due to incorrect shape {res.shape}")
                
    X = np.array(sequences)
    
    # Convert labels into one-hot encoded vectors
    y = to_categorical(labels, num_classes=num_classes).astype(int)
    
    return X, y, actions, num_classes

def main():
    print("Loading data...")
    X, y, actions, num_classes = load_data(DATA_DIR)
    
    if len(X) == 0:
        print("Error: No data found. Please check your data directory.")
        return
        
    print(f"Total samples loaded: {X.shape[0]}")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Partitioning:
    # We want exactly 80% Training, 10% Validation, and 10% Testing with proportional class representation.
    
    # First, split the dataset into 80% Training and 20% Temporary (Validation + Testing)
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    except ValueError:
        print("Warning: Not enough data for an initial stratified split. Falling back to random split...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Next, split the 20% into halves (which yields 10% Validation and 10% Testing of total data)
    try:
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    except ValueError:
        print("Warning: Not enough data for a stratified validation/test split. Falling back to random split...")
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")

    # Build the Sequential GRU Model Architecture
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES_DIM)),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define the 3 required Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )

    # Train Model (Dynamic Training Strategy for Max 150 Epochs)
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # Final Evaluation
    print("\nEvaluating model on Testing Set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
