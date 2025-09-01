# model_training.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------- Data Loading & Preprocessing -----------------
def load_and_preprocess(path):
    df = pd.read_csv(path, low_memory=False)

    # Drop unnamed/constant columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.apply(pd.Series.nunique) > 1]
    df.dropna(how='all', inplace=True)

    # Find target column
    possible_targets = ['label', 'Label', 'attack_cat', 'Attack_cat',
                        'class', 'Class', 'target', 'Target', 'Output']
    target_col = next((col for col in df.columns if col.strip() in possible_targets), None)
    if target_col is None:
        raise ValueError("❌ Target column not found. Please ensure dataset has a label column.")

    # Encode target labels
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])
    y = to_categorical(y)

    # Drop non-numeric + target columns
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    # Fill missing values
    X.fillna(0, inplace=True)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for Conv1D input
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    return X_scaled, y, scaler, label_encoder


# ----------------- Model Definition -----------------
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ----------------- Training + Evaluation -----------------
def train_model(path, model_name="intrusion_model.h5"):
    print(f"📥 Loading dataset from: {path}")
    X, y, scaler, label_encoder = load_and_preprocess(path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build model
    model = build_model(X_train.shape[1:], y.shape[1])

    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor="val_loss",        # watch validation loss
        patience=3,                # stop if no improvement for 3 epochs
        restore_best_weights=True  # roll back to best model
    )

    checkpoint = ModelCheckpoint(
        filepath=f"models/{model_name}",  # auto-save best model
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Train
    print("🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop, checkpoint] 
    )

    # Evaluate
    print("\n📊 Evaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # Classification Report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    print("\n🔹 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}")
    print(f"\n💾 Model saved as: models/{model_name}")
