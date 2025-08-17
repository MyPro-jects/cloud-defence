# main_intrusion.py

# 🔹 Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# 🔹 Function: Load dataset from CSV (URL or Drive)
def load_dataset(path, column_names=None):
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, names=column_names) if column_names else pd.read_csv(path)
    return df

# 🔹 Function: Preprocess dataset
def preprocess_data(df, label_column='label'):
    df = df.dropna()  # Drop rows with missing values
    
    # Encode labels (e.g., 'attack' → 1, 'normal' → 0)
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Normalize feature values between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for Conv1D input (samples, features, channels)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    return X_scaled, y, scaler, label_encoder

# 🔹 Function: Build Hybrid Conv1D + BiLSTM Model
def build_model(input_shape):
    model = Sequential([
        # Conv1D: learns local patterns in features
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        
        # MaxPooling: reduces dimensions
        MaxPooling1D(pool_size=2),
        
        # BiLSTM: learns temporal relationships in both directions
        Bidirectional(LSTM(64)),
        
        # Dropout for regularization
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output: sigmoid activation for binary classification (attack vs normal)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model with loss and optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 🔹 Function: Train model and save to .h5 file
def train_model(path, column_names=None, label_column='label', model_name='model_trained.h5'):
    # Step 1: Load and preprocess dataset
    df = load_dataset(path, column_names)
    X, y, scaler, encoder = preprocess_data(df, label_column)

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Build and train the model
    model = build_model(input_shape=(X_train.shape[1], 1))
    print("[INFO] Training the model...")
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    # Step 4: Save the model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{model_name}")
    print(f"[✅] Model saved as: models/{model_name}")

# 🔹 Function: Load trained model and predict from input sample
def predict(sample, model_path):
    model = load_model(model_path)
    sample = np.array(sample).reshape(1, -1, 1)  # Reshape for prediction
    pred = model.predict(sample)
    return pred
