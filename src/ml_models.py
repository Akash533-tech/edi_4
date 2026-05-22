import os
# Force TensorFlow to run on CPU and avoid macOS Metal initialization hangs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Prevent thread pool hangs on Apple Silicon under background runners
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
# Allow duplicate OpenMP runtime libraries to prevent deadlocks
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
# Configure TensorFlow to only use CPU
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time

def train_and_evaluate():
    processed_dir = "/Users/akashsunilsomsetwar/Desktop/edi_4/data_processed"
    
    # Load tabular features
    df_features = pd.read_csv(os.path.join(processed_dir, "cycle_features.csv"))
    
    # Load LSTM sequences
    lstm_sequences = np.load(os.path.join(processed_dir, "lstm_sequences.npy"))
    df_lstm_targets = pd.read_csv(os.path.join(processed_dir, "lstm_targets.csv"))
    df_lstm_meta = pd.read_csv(os.path.join(processed_dir, "lstm_meta.csv"))
    
    # 1. Define Train/Test split by Battery ID for cross-battery cross-condition evaluation
    # We train on 9 batteries and test on 3 batteries (one of each condition)
    test_batteries = ['RW12', 'RW20', 'RW28']
    
    # Create masks
    train_mask = ~df_features['battery_id'].isin(test_batteries)
    test_mask = df_features['battery_id'].isin(test_batteries)
    
    # --- Tabular Data Split for Random Forest ---
    features_cols = ['cycle', 'ri', 'peak_temp', 'voltage_drop', 'duration', 'mean_rw_temp', 'max_rw_temp']
    
    X_train_rf = df_features.loc[train_mask, features_cols].values.astype(np.float32)
    y_train_soh = df_features.loc[train_mask, 'soh'].values.astype(np.float32)
    y_train_rul = df_features.loc[train_mask, 'rul'].values.astype(np.float32)
    
    X_test_rf = df_features.loc[test_mask, features_cols].values.astype(np.float32)
    y_test_soh = df_features.loc[test_mask, 'soh'].values.astype(np.float32)
    y_test_rul = df_features.loc[test_mask, 'rul'].values.astype(np.float32)
    
    # Keep track of meta for plotting
    test_meta = df_features.loc[test_mask, ['battery_id', 'condition', 'cycle']].copy()
    
    # --- Sequence Data Split for LSTM ---
    # Scaling sequence inputs per channel
    # Sequence shape: (N, 100, 3) where columns are [Voltage, Current, Temperature]
    # We'll normalize manually based on training set statistics
    seq_mean = np.mean(lstm_sequences[train_mask], axis=(0, 1))
    seq_std = np.std(lstm_sequences[train_mask], axis=(0, 1))
    
    # Avoid division by zero
    seq_std[seq_std == 0] = 1.0
    
    lstm_sequences_scaled = (lstm_sequences - seq_mean) / seq_std
    
    X_train_lstm = lstm_sequences_scaled[train_mask].astype(np.float32)
    X_test_lstm = lstm_sequences_scaled[test_mask].astype(np.float32)
    
    # 2. Train Random Forest Models
    print("Training Random Forest models...")
    rf_soh = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
    rf_rul = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
    
    rf_soh.fit(X_train_rf, y_train_soh)
    rf_rul.fit(X_train_rf, y_train_rul)
    
    # Predict RF
    y_pred_rf_soh = rf_soh.predict(X_test_rf)
    y_pred_rf_rul = rf_rul.predict(X_test_rf)
    
    # Save RF models and scaler parameters
    with open(os.path.join(processed_dir, "rf_soh.pkl"), "wb") as f:
        pickle.dump(rf_soh, f)
    with open(os.path.join(processed_dir, "rf_rul.pkl"), "wb") as f:
        pickle.dump(rf_rul, f)
        
    # 3. Train LSTM Models
    print("Training LSTM models...")
    
    # SOH LSTM
    model_soh = Sequential([
        Input(shape=(100, 3)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model_soh.compile(optimizer='adam', loss='mse')
    
    # RUL LSTM
    model_rul = Sequential([
        Input(shape=(100, 3)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model_rul.compile(optimizer='adam', loss='mse')
    
    import sys
    class SimpleProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, target_name):
            self.target_name = target_name
        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get('loss', 0.0)
            val_loss = logs.get('val_loss', 0.0)
            print(f"  {self.target_name} - Epoch {epoch+1:02d}/25: loss={loss:.4f}, val_loss={val_loss:.4f}")
            sys.stdout.flush()

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    print("Fitting LSTM SOH...")
    model_soh.fit(
        X_train_lstm, y_train_soh,
        epochs=25,
        batch_size=16,
        validation_split=0.15,
        callbacks=[early_stop, SimpleProgressCallback("SOH")],
        verbose=0
    )
    
    print("Fitting LSTM RUL...")
    model_rul.fit(
        X_train_lstm, y_train_rul,
        epochs=25,
        batch_size=16,
        validation_split=0.15,
        callbacks=[early_stop, SimpleProgressCallback("RUL")],
        verbose=0
    )
    
    # Predict LSTM
    y_pred_lstm_soh = model_soh.predict(X_test_lstm).flatten()
    y_pred_lstm_rul = model_rul.predict(X_test_lstm).flatten()
    
    # Save LSTM models and normalization params
    model_soh.save(os.path.join(processed_dir, "lstm_soh.keras"))
    model_rul.save(os.path.join(processed_dir, "lstm_rul.keras"))
    
    norm_params = {'mean': seq_mean, 'std': seq_std}
    with open(os.path.join(processed_dir, "seq_norm_params.pkl"), "wb") as f:
        pickle.dump(norm_params, f)
        
    # 4. Calculate Metrics
    metrics = []
    
    # RF SOH
    rf_soh_rmse = np.sqrt(mean_squared_error(y_test_soh, y_pred_rf_soh))
    rf_soh_mae = mean_absolute_error(y_test_soh, y_pred_rf_soh)
    rf_soh_r2 = r2_score(y_test_soh, y_pred_rf_soh)
    metrics.append({'Model': 'Random Forest', 'Target': 'SOH', 'RMSE': rf_soh_rmse, 'MAE': rf_soh_mae, 'R2': rf_soh_r2})
    
    # RF RUL
    rf_rul_rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred_rf_rul))
    rf_rul_mae = mean_absolute_error(y_test_rul, y_pred_rf_rul)
    rf_rul_r2 = r2_score(y_test_rul, y_pred_rf_rul)
    metrics.append({'Model': 'Random Forest', 'Target': 'RUL', 'RMSE': rf_rul_rmse, 'MAE': rf_rul_mae, 'R2': rf_rul_r2})
    
    # LSTM SOH
    lstm_soh_rmse = np.sqrt(mean_squared_error(y_test_soh, y_pred_lstm_soh))
    lstm_soh_mae = mean_absolute_error(y_test_soh, y_pred_lstm_soh)
    lstm_soh_r2 = r2_score(y_test_soh, y_pred_lstm_soh)
    metrics.append({'Model': 'LSTM', 'Target': 'SOH', 'RMSE': lstm_soh_rmse, 'MAE': lstm_soh_mae, 'R2': lstm_soh_r2})
    
    # LSTM RUL
    lstm_rul_rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred_lstm_rul))
    lstm_rul_mae = mean_absolute_error(y_test_rul, y_pred_lstm_rul)
    lstm_rul_r2 = r2_score(y_test_rul, y_pred_lstm_rul)
    metrics.append({'Model': 'LSTM', 'Target': 'RUL', 'RMSE': lstm_rul_rmse, 'MAE': lstm_rul_mae, 'R2': lstm_rul_r2})
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(processed_dir, "model_comparison_metrics.csv"), index=False)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(df_metrics.to_string(index=False))
    
    # Save predictions for the test set
    test_meta['y_actual_soh'] = y_test_soh
    test_meta['y_pred_rf_soh'] = y_pred_rf_soh
    test_meta['y_pred_lstm_soh'] = y_pred_lstm_soh
    
    test_meta['y_actual_rul'] = y_test_rul
    test_meta['y_pred_rf_rul'] = y_pred_rf_rul
    test_meta['y_pred_lstm_rul'] = y_pred_lstm_rul
    
    test_meta.to_csv(os.path.join(processed_dir, "test_predictions.csv"), index=False)
    print(f"\nSaved test set predictions to test_predictions.csv")

if __name__ == "__main__":
    train_and_evaluate()
