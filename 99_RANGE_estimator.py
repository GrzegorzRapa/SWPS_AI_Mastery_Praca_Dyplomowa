import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import time
import gc
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

tf.config.threading.set_inter_op_parallelism_threads(4)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_DIR    = "driving_cycle_results"
BASE_OUTPUT_DIR = "trained_network_range_estimation_500"
TEMP_DIR    = r"D:\TEMP"

SELECTED_FILES_TRAINING = [
    "cycle_soc100_-20degC_generated_cycle_001_ECO.csv",
    "cycle_soc100_-20degC_generated_cycle_002_NORMAL.csv",
    "cycle_soc100_-20degC_generated_cycle_003_AGGRESIVE.csv",
    "cycle_soc100_0degC_generated_cycle_005_ECO.csv",
    "cycle_soc100_0degC_generated_cycle_006_NORMAL.csv",
    "cycle_soc100_0degC_generated_cycle_007_AGGRESIVE.csv",
    "cycle_soc100_20degC_generated_cycle_008_ECO.csv",
    "cycle_soc100_20degC_generated_cycle_009_NORMAL.csv",
    "cycle_soc100_20degC_generated_cycle_011_AGGRESIVE.csv"
    ]
    

SELECTED_FILES_VALIDATION = [
    "cycle_soc100_-20degC_generated_cycle_005_AGGRESIVE.csv",
    "cycle_soc100_-20degC_generated_cycle_007_ECO.csv",
    "cycle_soc100_-20degC_generated_cycle_009_NORMAL.csv",
    "cycle_soc100_20degC_generated_cycle_002_AGGRESIVE.csv",   
    "cycle_soc100_0degC_generated_cycle_009_ECO.csv",
    "cycle_soc100_20degC_generated_cycle_012_ECO.csv"
]


FEATURE_COLS = [
    "Battery_Voltage[V]", 
    "Battery_Current[A]", 
    "Battery_Temp[degC]",
    "Vehicle_Speed[km/h]"
]
TARGET_COL   = "Remaining_Distance[km]"

MODEL_CONFIGS = {
    "CNN": {
        "WINDOW_SIZE": 500, "BATCH_SIZE": 64, "EPOCHS": 10, "LEARNING_RATE": 1e-3, 
    },
    "LSTM": {
        "WINDOW_SIZE": 500, "BATCH_SIZE": 32, "EPOCHS": 20, "LEARNING_RATE": 1e-3,
    },
    "CNN_LSTM": {
        "WINDOW_SIZE": 500, "BATCH_SIZE": 32, "EPOCHS": 20, "LEARNING_RATE": 1e-3,
    }
}

# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================

def super_safe_read(path):
    try:
        return pd.read_csv(path, sep=";", encoding='cp1250', engine='python', on_bad_lines='skip')
    except:
        try:
            return pd.read_csv(path, sep=";", encoding='utf-8', errors='replace', engine='python')
        except Exception as e:
            raise ValueError(f"Critical error reading file {path}: {e}")

def get_next_version_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    next_id = 0 if not existing else max(int(d) for d in existing) + 1
    version_dir = os.path.join(base_dir, f"{next_id:03d}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir

def get_windows(file_list, scaler_x, scaler_y, window_size):
    """Extracts sequential windows. Both X and y are scaled."""
    Xs, ys, indices = [], [], []
    current_offset = 0
    boundaries = {}
    
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path): continue
        
        df = super_safe_read(path)
        X_raw = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype("float32")
        y_raw = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).values.astype("float32").reshape(-1, 1)
        
        X_scaled = scaler_x.transform(X_raw)
        y_scaled = scaler_y.transform(y_raw).flatten()
        
        n_win = len(X_scaled) // window_size
        start_idx_in_array = len(Xs)
        
        for i in range(n_win):
            start = i * window_size
            end = start + window_size
            Xs.append(X_scaled[start:end])
            ys.append(y_scaled[end-1])
            indices.append(current_offset + end)
            
        end_idx_in_array = len(Xs)
        boundaries[fname] = (start_idx_in_array, end_idx_in_array)
        current_offset += len(df)
        
    return np.array(Xs), np.array(ys), np.array(indices), boundaries

# =============================================================================
# 3. MODELS & TOOLS
# =============================================================================

def build_model(name, input_shape, lr):
    if name == "CNN":
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
    elif name == "LSTM":
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64),
            layers.Dense(1)
        ])
    else: # CNN_LSTM
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, activation="relu"),
            layers.LSTM(32),
            layers.Dense(1)
        ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mse"])
    return m

def plot_results(results_dict, output_path, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    ref_key = list(results_dict.keys())[0]
    ax1.plot(results_dict[ref_key]["indices"], results_dict[ref_key]["y_val_km"], 'k', label='Real Range', alpha=0.5)
    
    for m_name, data in results_dict.items():
        ax1.plot(data["indices"], data["y_pred_km"], '--', label=f'Pred {m_name}', alpha=0.8)
    
    ax1.set_title(f"Range Estimation [km] {title_suffix}")
    ax1.set_ylabel("Distance [km]")
    ax1.legend()
    ax1.grid(True, linestyle='--')
    
    for m_name, data in results_dict.items():
        error = data["y_val_km"] - data["y_pred_km"]
        ax2.plot(data["indices"], error, label=f'Error {m_name}', alpha=0.7)
        
    ax2.set_ylabel("Error [km]")
    ax2.axhline(0, color='k')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def convert_to_tflite(keras_model, output_path, precision):
    """Converts a model to TFLite format by unrolling RNNs to avoid TensorListReserve."""
    print(f"[TFLITE] Converting model to {precision}...")
    
    # Tworzymy nowy model ze statycznym kształtem wejścia
    static_model = tf.keras.Sequential()
    static_model.add(tf.keras.layers.InputLayer(batch_input_shape=(1, *keras_model.input_shape[1:])))
    
    for layer in keras_model.layers:
        config = layer.get_config()
        
        # KLUCZOWA ZMIANA: Zmuszamy warstwy LSTM do "rozwinięcia" (unroll).
        if 'LSTM' in layer.__class__.__name__:
            config['unroll'] = True
            
        new_layer = layer.__class__.from_config(config)
        static_model.add(new_layer)
        
    # Kopiujemy wagi do nowego, statycznego i rozwiniętego modelu
    static_model.set_weights(keras_model.get_weights())

    # Konwersja TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Wymuszamy tylko wbudowane, bezproblemowe operacje
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    if precision == "16-bit":
        converter.target_spec.supported_types = [tf.float16]
    elif precision == "8-bit":
        # Dynamic Range Quantization (domyślne 8-bit)
        pass
        
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[TFLITE] Saved: {output_path}")
    

def predict_with_tflite(tflite_path, X_val):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    preds = []
    for i in range(len(X_val)):
        inp = np.expand_dims(X_val[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        preds.append(interpreter.get_tensor(output_details['index'])[0][0])
    return np.array(preds)

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def main():
    session_dir = get_next_version_dir(BASE_OUTPUT_DIR)
    
    # 1. Fit Scalers
    all_x, all_y = [], []
    for f in SELECTED_FILES_TRAINING:
        df = super_safe_read(os.path.join(DATA_DIR, f))
        all_x.append(df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values)
        all_y.append(df[TARGET_COL].apply(pd.to_numeric, errors='coerce').fillna(0).values.reshape(-1, 1))
    
    scaler_x = MinMaxScaler().fit(np.concatenate(all_x, axis=0))
    scaler_y = MinMaxScaler().fit(np.concatenate(all_y, axis=0))
    
    joblib.dump(scaler_x, os.path.join(session_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(session_dir, "scaler_y.pkl"))

    results_keras, results_t16, results_t8 = {}, {}, {}
    training_histories = {} # ZMIANA: Dodano słownik do przechowywania historii uczenia
    report = ["=== RANGE TRAINING REPORT (SCALED Y) ===\n"]

    for m_name, cfg in MODEL_CONFIGS.items():
        ws = cfg["WINDOW_SIZE"]
        X_t, y_t, _, _ = get_windows(SELECTED_FILES_TRAINING, scaler_x, scaler_y, ws)
        X_v, y_v, idx_v, val_bounds = get_windows(SELECTED_FILES_VALIDATION, scaler_x, scaler_y, ws)

        model = build_model(m_name, (ws, len(FEATURE_COLS)), cfg["LEARNING_RATE"])
        history = model.fit(X_t, y_t, epochs=cfg["EPOCHS"], batch_size=cfg["BATCH_SIZE"], verbose=1)
        
        training_histories[m_name] = history.history # ZMIANA: Zapis historii dla każdego modelu
        
        sciezka_zapisu = os.path.join(session_dir, f"{m_name}_unquantized.keras")
        model.save(sciezka_zapisu)
        print(f"Zapisano nieskwantowany model w: {sciezka_zapisu}")



        # --- Keras Predict & Descale ---
        y_p_raw = model.predict(X_v, verbose=0).flatten()
        # Inwersja skalowania dla wyników
        y_v_km = scaler_y.inverse_transform(y_v.reshape(-1, 1)).flatten()
        y_p_km = scaler_y.inverse_transform(y_p_raw.reshape(-1, 1)).flatten()
        results_keras[m_name] = {"y_val_km": y_v_km, "y_pred_km": y_p_km, "indices": idx_v}

        # --- TFLite 16-bit Predict & Descale ---
        path_16 = os.path.join(session_dir, f"{m_name}_16.tflite")
        convert_to_tflite(model, path_16, "16-bit")
        y_p16_raw = predict_with_tflite(path_16, X_v)
        y_p16_km = scaler_y.inverse_transform(y_p16_raw.reshape(-1, 1)).flatten()
        results_t16[m_name] = {"y_val_km": y_v_km, "y_pred_km": y_p16_km, "indices": idx_v}

        # --- TFLite 8-bit Predict & Descale ---
        path_8 = os.path.join(session_dir, f"{m_name}_8.tflite")
        convert_to_tflite(model, path_8, "8-bit")
        y_p8_raw = predict_with_tflite(path_8, X_v)
        y_p8_km = scaler_y.inverse_transform(y_p8_raw.reshape(-1, 1)).flatten()
        results_t8[m_name] = {"y_val_km": y_v_km, "y_pred_km": y_p8_km, "indices": idx_v}

        # Log errors in KM
        report.append(f"Model: {m_name}")
        for fname, (s, e) in val_bounds.items():
            if e > s:
                err = np.max(np.abs(y_v_km[s:e] - y_p_km[s:e]))
                report.append(f" - {fname}: Max Err {err:.2f} km")
        report.append("\n")
        
        gc.collect()

    # Save Plots & Report
    plot_results(results_keras, os.path.join(session_dir, "plot_km.png"), "(Keras)")
    plot_results(results_t16, os.path.join(session_dir, "plot_tflite_16_km.png"), "(TFLite 16bit)")
    plot_results(results_t8, os.path.join(session_dir, "plot_tflite_8_km.png"), "(TFLite 8bit)")
    
    # ZMIANA: Rysowanie wykresu historii uczenia (Loss i MSE)
    fig_hist, (ax_loss, ax_mse) = plt.subplots(1, 2, figsize=(14, 6))
    for m_name, hist in training_histories.items():
        epochs_range = range(1, len(hist['loss']) + 1)
        ax_loss.plot(epochs_range, hist['loss'], marker='o', label=f'{m_name}')
        ax_mse.plot(epochs_range, hist['mse'], marker='x', label=f'{m_name}')
    
    ax_loss.set_title("Training Loss across Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)
    
    ax_mse.set_title("Training MSE across Epochs")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE")
    ax_mse.legend()
    ax_mse.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, "training_history.png"))
    plt.close()
    
    with open(os.path.join(session_dir, "report.txt"), "w") as f: f.write("\n".join(report))
    print(f"Finished. Results in: {session_dir}")

if __name__ == "__main__":
    main()