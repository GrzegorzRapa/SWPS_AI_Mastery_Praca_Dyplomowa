import os
# Enable oneDNN optimizations for Intel/AMD CPUs before loading TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
# Optimize CPU threads usage (limit to 4 cores)

import time
import gc
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
BASE_OUTPUT_DIR = "tranined_network_soc_estimation_500"
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
    "Acceleration[m/s2]", 
    "Battery_Voltage[V]", 
    "Battery_Current[A]", 
    "Battery_Temp[degC]"
]
TARGET_COL   = "Battery SoC[%]"

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
    """The most robust method for reading CSV files, handling Polish characters."""
    try:
        # Try 1: Windows-1250 (common in Polish Excel)
        return pd.read_csv(path, sep=";", encoding='cp1250', engine='python', on_bad_lines='skip')
    except:
        try:
            # Try 2: UTF-8 with error replacement
            return pd.read_csv(path, sep=";", encoding='utf-8', errors='replace', engine='python')
        except Exception as e:
            raise ValueError(f"Critical error reading file {path}: {e}")

def get_next_version_dir(base_dir):
    """Creates a new folder with an incremented ID (e.g., 001, 002)."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    next_id = 0 if not existing else max(int(d) for d in existing) + 1
    version_dir = os.path.join(base_dir, f"{next_id:03d}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir

def get_windows(file_list, scaler, window_size):
    """Extracts sequential windows from the data. Also returns file boundaries for per-file error calc."""
    Xs, ys, indices = [], [], []
    current_offset = 0
    boundaries = {} # format: {filename: (start_index, end_index)}
    
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path): 
            print(f"[WARN] File not found: {fname}")
            continue
        
        df = super_safe_read(path)
        # Ensure columns are numeric
        X_raw = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype("float32")
        y_raw = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).values.astype("float32")
        
        X_scaled = scaler.transform(X_raw)
        
        n_win = len(X_scaled) // window_size
        start_idx_in_array = len(Xs)
        
        for i in range(n_win):
            start = i * window_size
            end = start + window_size
            Xs.append(X_scaled[start:end])
            ys.append(y_raw[end-1])
            indices.append(current_offset + end)
            
        end_idx_in_array = len(Xs)
        boundaries[fname] = (start_idx_in_array, end_idx_in_array)
        current_offset += len(df)
        
    return np.array(Xs), np.array(ys), np.array(indices), boundaries

# =============================================================================
# 3. MODEL BUILDING, PLOTTING AND TFLITE
# =============================================================================

def build_model(name, input_shape, lr):
    """Builds the Keras model based on the selected architecture."""
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
    # Added 'mse' to metrics to track it explicitly in history
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss="mse", metrics=["mse"])
    return m

def plot_results(results_dict, output_path, title_suffix=""):
    """Creates a subplot with SoC predictions (top) and Error (bottom) with appropriate grids."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Górny Wykres: Predykcja SoC ---
    # Rysujemy prawdziwy przebieg SoC tylko raz (biorąc dane z pierwszego modelu z brzegu),
    # aby uniknąć nakładających się na siebie czarnych linii.
    ref_key = list(results_dict.keys())[0]
    ax1.plot(results_dict[ref_key]["indices"], results_dict[ref_key]["y_val"], 'k', label='Real SoC', alpha=0.5, linewidth=2)
    
    for model_name, data in results_dict.items():
        ax1.plot(data["indices"], data["y_pred"], '--', label=f'Pred {model_name}', alpha=0.8)
    
    ax1.set_title(f"SoC Estimation Validation {title_suffix}")
    ax1.set_ylabel("Battery SoC [%]")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5)) # Siatka co 5 punktów SoC
    
    # --- Dolny Wykres: Błąd Predykcji ---
    for model_name, data in results_dict.items():
        # KLUCZOWA ZMIANA: Obliczamy błąd używając y_val PRZYPISANEGO DO DANEGO MODELU.
        # Dzięki temu wymiary (długości list) y_val i y_pred zawsze idealnie do siebie pasują!
        error = data["y_val"] - data["y_pred"]
        ax2.plot(data["indices"], error, label=f'Error {model_name}', alpha=0.7)
        
    ax2.set_xlabel("Data points (Time)")
    ax2.set_ylabel("Error (Real - Pred) [%]")
    ax2.axhline(0, color='k', linewidth=1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1)) # Siatka co 1 punkt błędu
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def convert_to_tflite(keras_model, output_path, precision, X_train=None):
    """Converts a Keras model to TFLite format natively (no Flex Ops, Edge friendly)."""
    print(f"[TFLITE] Converting model to {precision}...")
    
    # --- OSTATECZNE ROZWIĄZANIE DLA LSTM ---
    # Zamiast tylko kopiować stare warstwy, klonujemy ich ustawienia, 
    # aby zmusić Kerasa do całkowitego przebudowania grafu matematycznego od zera,
    # tym razem ze świadomością, że batch_size to zawsze 1.
    static_model = tf.keras.Sequential()
    static_model.add(tf.keras.layers.InputLayer(batch_input_shape=(1, *keras_model.input_shape[1:])))
    
    for layer in keras_model.layers:
        # Pobieramy konfigurację i tworzymy zupełnie nową instancję warstwy
        config = layer.get_config()
        new_layer = layer.__class__.from_config(config)
        static_model.add(new_layer)
        
    # Kopiujemy wyuczone wagi z oryginalnego modelu
    static_model.set_weights(keras_model.get_weights())

    # Konwersja
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Wymuszamy TYLKO wbudowane, lekkie operacje TFLite (bez Flex Ops!)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    if precision == "16-bit":
        converter.target_spec.supported_types = [tf.float16]
        
    elif precision == "8-bit":
        # 8-bit korzysta z metody "Dynamic Range Quantization", 
        # która jest domyślna, lekka i świetnie radzi sobie z sieciami LSTM na małych urządzeniach.
        pass
        
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[TFLITE] Saved: {output_path}")


def predict_with_tflite(tflite_path, X_val):
    """Runs inference on validation data using a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    is_int8 = input_details['dtype'] == np.int8
    if is_int8:
        in_scale, in_zp = input_details['quantization']
        out_scale, out_zp = output_details['quantization']
        
    predictions = []
    # Predict sample by sample
    for i in range(len(X_val)):
        input_data = np.expand_dims(X_val[i], axis=0).astype(np.float32)
        
        # Quantize input if 8-bit model
        if is_int8:
            input_data = (input_data / in_scale + in_zp).astype(np.int8)
            
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])[0][0]
        
        # Dequantize output if 8-bit model
        if is_int8:
            output_data = (np.float32(output_data) - out_zp) * out_scale
            
        predictions.append(output_data)
        
    return np.array(predictions)

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def main():
    session_dir = get_next_version_dir(BASE_OUTPUT_DIR)
    print(f"[SESSION] Output directory: {session_dir}")

    try:
        print("[INFO] Fitting MinMaxScaler on training data...")
        all_train_raw = []
        for f in SELECTED_FILES_TRAINING:
            df = super_safe_read(os.path.join(DATA_DIR, f))
            all_train_raw.append(df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values)
        
        scaler = MinMaxScaler().fit(np.concatenate(all_train_raw, axis=0))
        joblib.dump(scaler, os.path.join(session_dir, "scaler.pkl"))

        # Dictionaries to hold results for regular and TFLite models
        results_keras = {}
        results_tflite_16 = {}
        results_tflite_8 = {}
        
        # Store training history and info
        training_histories = {}
        report_info = []

        report_info.append("=== SOC ESTIMATION TRAINING REPORT ===\n")

        for m_name, cfg in MODEL_CONFIGS.items():
            print(f"\n--- PREPARING MODEL: {m_name} ---")
            win_size = cfg["WINDOW_SIZE"]
            
            # Load Data
            X_t, y_t, _, _ = get_windows(SELECTED_FILES_TRAINING, scaler, win_size)
            X_v, y_v, idx_v, val_boundaries = get_windows(SELECTED_FILES_VALIDATION, scaler, win_size)
            
            # Add info to report
            if m_name == list(MODEL_CONFIGS.keys())[0]:
                report_info.append(f"Features: {FEATURE_COLS}\n")
                report_info.append(f"Scaler parameters (Min): {scaler.data_min_}\n")
                report_info.append(f"Scaler parameters (Max): {scaler.data_max_}\n\n")

            report_info.append(f"--- MODEL: {m_name} ---")
            report_info.append(f"Training vectors count: {len(X_t)}")
            report_info.append(f"Vector (Window) length: {win_size}")
            
            # Create Dataset
            ds = tf.data.Dataset.from_tensor_slices((X_t, y_t))
            ds = ds.shuffle(len(X_t)).batch(cfg["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
            
            model = build_model(m_name, (win_size, len(FEATURE_COLS)), cfg["LEARNING_RATE"])
            
            # Record model summary into the report
            report_info.append("Model Structure:")
            model.summary(print_fn=lambda x: report_info.append(x))
            
            print(f"--- STARTING TRAINING: {m_name} ---")
            start_time = time.time()
            history = model.fit(ds, epochs=cfg["EPOCHS"], verbose=1)
            train_time = time.time() - start_time
            
            training_histories[m_name] = history.history
            report_info.append(f"Training time: {train_time:.2f} seconds")
            
            # Log epoch metrics
            report_info.append("Metrics per epoch:")
            for e in range(cfg["EPOCHS"]):
                report_info.append(f" Epoch {e+1}: Loss = {history.history['loss'][e]:.6f}, MSE = {history.history['mse'][e]:.6f}")
            
            # Save standard Keras model
            keras_model_path = os.path.join(session_dir, f"{m_name.lower()}.keras")
            model.save(keras_model_path)
            
            # Keras Prediction & Error tracking
            y_pred = model.predict(X_v, verbose=0).flatten()
            results_keras[m_name] = {"y_val": y_v, "y_pred": y_pred, "indices": idx_v}
            
            report_info.append("Max errors per validation file (Keras):")
            for fname, (s_idx, e_idx) in val_boundaries.items():
                if e_idx > s_idx: # Ensure file had valid windows
                    file_error = np.abs(y_v[s_idx:e_idx] - y_pred[s_idx:e_idx])
                    report_info.append(f" - {fname}: {np.max(file_error):.2f}%")
            report_info.append("\n")

            # --- TFLITE 16-bit Conversion & Prediction ---
            tflite_16_path = os.path.join(session_dir, f"{m_name.lower()}_16bit.tflite")
            convert_to_tflite(model, tflite_16_path, "16-bit")
            y_pred_16 = predict_with_tflite(tflite_16_path, X_v)
            results_tflite_16[m_name] = {"y_val": y_v, "y_pred": y_pred_16, "indices": idx_v}

            # --- TFLITE 8-bit Conversion & Prediction ---
            tflite_8_path = os.path.join(session_dir, f"{m_name.lower()}_8bit.tflite")
            convert_to_tflite(model, tflite_8_path, "8-bit", X_train=X_t)
            y_pred_8 = predict_with_tflite(tflite_8_path, X_v)
            results_tflite_8[m_name] = {"y_val": y_v, "y_pred": y_pred_8, "indices": idx_v}

            keras.backend.clear_session()
            gc.collect()

        # =====================================================================
        # 5. GENERATE ALL OUTPUT FILES (PLOTS & REPORT)
        # =====================================================================
        
        print("\n[INFO] Generating plots and reports...")
        
        # Plot 1: Keras models
        plot_results(results_keras, os.path.join(session_dir, "plot.png"), "(Standard Keras)")
        # Plot 2: TFLite 16-bit
        plot_results(results_tflite_16, os.path.join(session_dir, "plot_tflite_16bit.png"), "(TFLite 16-bit Float)")
        # Plot 3: TFLite 8-bit
        plot_results(results_tflite_8, os.path.join(session_dir, "plot_tflite_8bit.png"), "(TFLite 8-bit Integer)")

        # Plot 4: Training History (Loss / MSE)
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

        # Save Text Report
        with open(os.path.join(session_dir, "training_info.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(report_info))

        print(f"[DONE] All tasks completed successfully. Check directory: {session_dir}")

    finally:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()