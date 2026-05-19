import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
import tensorflow as tf
from tensorflow import keras

# Suppress scikit-learn version warnings for the MinMaxScaler
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Enable oneDNN optimizations for Intel/AMD CPUs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# Optimize CPU threads usage
tf.config.threading.set_inter_op_parallelism_threads(4)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

FOLDER_NUMBER = "000" 

DATA_DIR = "driving_cycle_results"
BASE_MODEL_DIR = "tranined_network_soc_estimation_100"
BASE_OUTPUT_DIR = "validation_soc_new_oneandhalf_window"

TRAINED_NETWORKS_DIR = os.path.join(BASE_MODEL_DIR, FOLDER_NUMBER)
VALIDATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, BASE_MODEL_DIR, FOLDER_NUMBER)

SELECTED_FILES_VALIDATION = [
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_001_NORMAL.csv", 
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_003_AGGRESIVE.csv",
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_007_ECO.csv", 
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_011_ECO.csv",
]

FEATURE_COLS = ["Acceleration[m/s2]", "Battery_Voltage[V]", "Battery_Current[A]", "Battery_Temp[degC]"]
TARGET_COL = "Battery SoC[%]"

MODEL_CONFIGS = {
    "cnn": {"window": 100, "keras": "cnn.keras", "tflite_16": "cnn_16bit.tflite", "tflite_8": "cnn_8bit.tflite"},
    "cnn_lstm": {"window": 100, "keras": "cnn_lstm.keras", "tflite_16": "cnn_lstm_16bit.tflite", "tflite_8": "cnn_lstm_8bit.tflite"},
    "lstm": {"window": 100, "keras": "lstm.keras", "tflite_16": "lstm_16bit.tflite", "tflite_8": "lstm_8bit.tflite"}
}

# New variable for warm-up analysis
WARMUP_WINDOW = 150

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def super_safe_read(path):
    """Robust CSV reader handling different encodings."""
    try:
        return pd.read_csv(path, sep=";", encoding='cp1250', engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(path, sep=";", encoding='utf-8', errors='replace', engine='python')

def get_windows_single_file(file_path, scaler, window_size, offset=0):
    """Prepares windowed data for a single file."""
    df = super_safe_read(file_path)
    X_raw = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype("float32")
    y_raw = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).values.astype("float32")
    X_scaled = scaler.transform(X_raw)
    
    Xs, ys, indices = [], [], []
    n_win = len(X_scaled) // window_size
    for i in range(n_win):
        start = i * window_size
        end = start + window_size
        Xs.append(X_scaled[start:end])
        ys.append(y_raw[end-1])
        indices.append(offset + end)
    return np.array(Xs), np.array(ys), np.array(indices)

def predict_with_tflite(tflite_path, X_val):
    """Inference for TFLite models (supports Float16 and Int8)."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]
    
    is_int8 = input_det['dtype'] == np.int8
    predictions = []
    for i in range(len(X_val)):
        input_data = np.expand_dims(X_val[i], axis=0).astype(np.float32)
        if is_int8:
            in_scale, in_zp = input_det['quantization']
            input_data = (input_data / in_scale + in_zp).astype(np.int8)
        interpreter.set_tensor(input_det['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_det['index'])[0][0]
        if is_int8:
            out_scale, out_zp = output_det['quantization']
            output_data = (np.float32(output_data) - out_zp) * out_scale
        predictions.append(output_data)
    return np.array(predictions)

def plot_single_network_validation(y_val, y_pred, indices, output_path, title):
    """Generates a comparison plot for SoC prediction and error."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(indices, y_val, 'k', label='Real SoC', alpha=0.6, linewidth=2)
    ax1.plot(indices, y_pred, 'r--', label='Predicted SoC', alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Battery SoC [%]")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    
    error = y_val - y_pred
    ax2.plot(indices, error, color='blue', label='Error', alpha=0.7)
    ax2.set_xlabel("Data points (Time)")
    ax2.set_ylabel("Error [%]")
    ax2.axhline(0, color='k', linewidth=1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_combined_validation(real_y, preds_dict, indices, output_path, title):
    """Generates a combined comparison plot for all models under a specific quantization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Colors matching the requested visual style
    colors = {"cnn": "tab:blue", "lstm": "tab:orange", "cnn_lstm": "tab:green"}
    
    ax1.plot(indices, real_y, color='gray', label='Real SoC', alpha=0.8, linewidth=2)
    
    for m_name, y_pred in preds_dict.items():
        label_name = f"Pred {m_name.upper()}"
        ax1.plot(indices, y_pred, linestyle='--', color=colors.get(m_name, 'black'), label=label_name, alpha=0.8)
        
    ax1.set_title(title)
    ax1.set_ylabel("Battery SoC [%]")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    
    for m_name, y_pred in preds_dict.items():
        error = np.array(real_y) - np.array(y_pred)
        label_name = f"Error {m_name.upper()}"
        ax2.plot(indices, error, color=colors.get(m_name, 'black'), label=label_name, alpha=0.6)
        
    ax2.set_xlabel("Data points (Time)")
    ax2.set_ylabel("Error (Real - Pred) [%]")
    ax2.axhline(0, color='k', linewidth=1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =============================================================================
# 3. MAIN PROCESS
# =============================================================================

def main():
    os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
    
    scaler_path = os.path.join(TRAINED_NETWORKS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler not found at: {scaler_path}")
        return
    scaler = joblib.load(scaler_path)
    
    report_dict = {}
    
    # Aggregation structure for combined plots
    combined_plots_data = {
        "keras": {"indices": [], "real": [], "preds": {}},
        "16bit": {"indices": [], "real": [], "preds": {}},
        "8bit": {"indices": [], "real": [], "preds": {}}
    }

    for m_name, config in MODEL_CONFIGS.items():
        versions = [
            ("keras", m_name + "_keras", os.path.join(TRAINED_NETWORKS_DIR, config["keras"]), "keras"),
            ("16bit", m_name + "_16bit", os.path.join(TRAINED_NETWORKS_DIR, config["tflite_16"]), "tflite"),
            ("8bit", m_name + "_8bit", os.path.join(TRAINED_NETWORKS_DIR, config["tflite_8"]), "tflite")
        ]
        
        for q_type, full_net_name, model_path, m_type in versions:
            if not os.path.exists(model_path):
                print(f"[SKIP] Model file not found: {model_path}")
                continue

            print(f"Validation: {full_net_name}...")
            report_dict[full_net_name] = {}
            
            all_y_v, all_y_p, all_idx = [], [], []
            current_offset = 0
            
            model_obj = None
            if m_type == "keras":
                try:
                    model_obj = keras.models.load_model(model_path, compile=False)
                except Exception as e:
                    print(f"[ERROR] Could not load model {full_net_name}: {e}")
                    continue

            for file_name in SELECTED_FILES_VALIDATION:
                f_path = os.path.join(DATA_DIR, file_name)
                if not os.path.exists(f_path):
                    continue
                
                X, y, idx = get_windows_single_file(f_path, scaler, config["window"], offset=current_offset)
                
                if m_type == "keras":
                    y_p = model_obj.predict(X, verbose=0).flatten()
                else:
                    y_p = predict_with_tflite(model_path, X)
                
                # Single file plotting
                f_base = file_name.replace(".csv", "")
                img_name = f"{f_base}_{full_net_name}.png"
                plot_single_network_validation(y, y_p, idx, os.path.join(VALIDATION_OUTPUT_DIR, img_name), 
                                               f"Validation: {full_net_name} | File: {file_name}")
                
                # Error analysis including warm-up logic
                error_arr = np.abs(y - y_p)
                warmup_size = min(WARMUP_WINDOW, len(error_arr))
                
                max_err_warmup = np.max(error_arr[:warmup_size]) if warmup_size > 0 else 0.0
                max_err_after = np.max(error_arr[warmup_size:]) if len(error_arr) > warmup_size else 0.0
                
                report_dict[full_net_name][file_name] = {
                    "warmup_err": max_err_warmup,
                    "after_err": max_err_after
                }
                
                all_y_v.extend(y)
                all_y_p.extend(y_p)
                all_idx.extend(idx)
                current_offset += (len(y) * config["window"])

            if all_y_v:
                plot_single_network_validation(np.array(all_y_v), np.array(all_y_p), np.array(all_idx), 
                                               os.path.join(VALIDATION_OUTPUT_DIR, f"GLOBAL_VALIDATION_{full_net_name}.png"),
                                               f"Global Validation: {full_net_name} (All Files)")
                
                # Store data for combined plots
                combined_plots_data[q_type]["preds"][m_name] = np.array(all_y_p)
                if len(combined_plots_data[q_type]["real"]) == 0:
                    combined_plots_data[q_type]["real"] = np.array(all_y_v)
                    combined_plots_data[q_type]["indices"] = np.array(all_idx)
            
            if model_obj:
                keras.backend.clear_session()

    # Generate combined global validation plots
    for q_type, data in combined_plots_data.items():
        if data["real"] is not [] and data["preds"]:
            plot_combined_validation(
                data["real"], 
                data["preds"], 
                data["indices"],
                os.path.join(VALIDATION_OUTPUT_DIR, f"COMBINED_GLOBAL_{q_type.upper()}.png"),
                f"SoC Estimation Validation ({q_type.upper()})"
            )

    # Generate the text report
    report_path = os.path.join(VALIDATION_OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("=== SOC VALIDATION REPORT (BY NETWORK) ===\n\n")
        f.write(f"WARM-UP WINDOW: {WARMUP_WINDOW} points\n\n")
        for net, files in report_dict.items():
            f.write(f"MODEL: {net.upper()}\n")
            f.write("-" * 80 + "\n")
            for f_name, errs in files.items():
                f.write(f"  - {f_name:50}\n")
                f.write(f"      Max Error (Warm-up): {errs['warmup_err']:.2f}%\n")
                f.write(f"      Max Error (After):   {errs['after_err']:.2f}%\n")
            f.write("\n")

    print(f"\n[DONE] Validation results saved in: {VALIDATION_OUTPUT_DIR}")

if __name__ == "__main__":
    main()