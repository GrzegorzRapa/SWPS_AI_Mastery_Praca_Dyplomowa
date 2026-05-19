import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
import tensorflow as tf
from tensorflow import keras

# Ignorowanie ostrzeżeń dotyczących wersji scikit-learn dla MinMaxScaler
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Włączenie optymalizacji oneDNN dla procesorów Intel/AMD
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# Optymalizacja użycia wątków procesora
tf.config.threading.set_inter_op_parallelism_threads(4)

# =============================================================================
# 1. KONFIGURACJA
# =============================================================================

FOLDER_NUMBER = "000" 

DATA_DIR = "driving_cycle_results"
BASE_MODEL_DIR = "trained_network_range_estimation_200"
BASE_OUTPUT_DIR = "validation_output_range_new"

TRAINED_NETWORKS_DIR = os.path.join(BASE_MODEL_DIR, FOLDER_NUMBER)
VALIDATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, BASE_MODEL_DIR, FOLDER_NUMBER)

SELECTED_FILES_VALIDATION = [
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_001_NORMAL.csv", 
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_003_AGGRESIVE.csv",
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_007_ECO.csv", 
    "additonal_test_cycle_soc50_0_0degC_generated_cycle_011_ECO.csv",
]

FEATURE_COLS = ["Battery_Voltage[V]", "Battery_Current[A]", "Battery_Temp[degC]", "Vehicle_Speed[km/h]"]
TARGET_COL = "Remaining_Distance[km]"

MODEL_CONFIGS = {
    "CNN": {"window": 200, "keras": "CNN_unquantized.keras", "tflite_16": "CNN_16.tflite", "tflite_8": "CNN_8.tflite"},
    "LSTM": {"window": 200, "keras": "LSTM_unquantized.keras", "tflite_16": "LSTM_16.tflite", "tflite_8": "LSTM_8.tflite"},
    "CNN_LSTM": {"window": 200, "keras": "CNN_LSTM_unquantized.keras", "tflite_16": "CNN_LSTM_16.tflite", "tflite_8": "CNN_LSTM_8.tflite"}
}

WARMUP_WINDOW = 401

# =============================================================================
# 2. FUNKCJE POMOCNICZE
# =============================================================================

def super_safe_read(path):
    try:
        return pd.read_csv(path, sep=";", encoding='cp1250', engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(path, sep=";", encoding='utf-8', errors='replace', engine='python')

def get_windows_single_file(file_path, scaler_x, scaler_y, window_size, offset=0):
    df = super_safe_read(file_path)
    X_raw = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype("float32")
    y_raw = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).values.astype("float32").reshape(-1, 1)
    
    X_scaled = scaler_x.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw).flatten()
    
    Xs, ys, indices = [], [], []
    n_win = len(X_scaled) // window_size
    for i in range(n_win):
        start = i * window_size
        end = start + window_size
        Xs.append(X_scaled[start:end])
        ys.append(y_scaled[end-1])
        indices.append(offset + end)
    return np.array(Xs), np.array(ys), np.array(indices)

def predict_with_tflite(tflite_path, X_val):
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

def plot_validation_results(y_val, y_pred, indices, output_path, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(indices, y_val, 'k', label='Real Range', alpha=0.6, linewidth=2)
    ax1.plot(indices, y_pred, 'r--', label='Predicted Range', alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Distance [km]")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    error = y_val - y_pred
    ax2.plot(indices, error, color='blue', label='Error', alpha=0.7)
    ax2.set_xlabel("Data points")
    ax2.set_ylabel("Error [km]")
    ax2.axhline(0, color='k', linewidth=1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- NOWA FUNKCJA DO ZBIORCZEGO WYKRESU ---
def plot_combined_validation_results(y_val, dict_y_pred, indices, output_path, title):
    """
    Odpowiednik MATLABowego nakładania wielu wykresów (hold on).
    Iteruje przez dostarczone w słowniku predykcje i rysuje je na wspólnych osiach.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Rysowanie wartości rzeczywistej
    ax1.plot(indices, y_val, color='gray', label='Real Range', alpha=0.8, linewidth=2.5)
    
    # Pula kolorów dla predykcji poszczególnych sieci
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    # Pętla nakładająca predykcje z różnych modeli (Subplot 1)
    for i, (model_name, y_pred) in enumerate(dict_y_pred.items()):
        color = colors[i % len(colors)]
        ax1.plot(indices, y_pred, '--', color=color, label=f'Pred {model_name}', alpha=0.7, linewidth=1.5)
        
    ax1.set_title(title)
    ax1.set_ylabel("Distance [km]")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Pętla nakładająca błędy z różnych modeli (Subplot 2)
    ax2.axhline(0, color='gray', linewidth=2) # Linia zera jako punkt odniesienia
    for i, (model_name, y_pred) in enumerate(dict_y_pred.items()):
        color = colors[i % len(colors)]
        error = y_val - y_pred  # Formuła błędu (Real - Pred)
        ax2.plot(indices, error, '-', color=color, label=f'Error {model_name}', alpha=0.6, linewidth=1.2)
        
    ax2.set_xlabel("Data points")
    ax2.set_ylabel("Error [km]")
    ax2.legend(loc="lower right")
    # Gęstsza siatka dla błędów
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yticks(np.arange(-25, 11, 2)) # Przykładowa podziałka Y, można usunąć dla autokalibracji
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# =============================================================================
# 3. PROCES GŁÓWNY
# =============================================================================

def main():
    os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
    
    scaler_x_path = os.path.join(TRAINED_NETWORKS_DIR, "scaler_x.pkl")
    scaler_y_path = os.path.join(TRAINED_NETWORKS_DIR, "scaler_y.pkl")
    
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print(f"[ERROR] Skalery nie znalezione w: {TRAINED_NETWORKS_DIR}")
        return
        
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    report_dict = {}

    # --- NOWE ZMIENNE DO ZBIERANIA DANYCH ZBIORCZYCH ---
    combined_keras_preds = {}
    combined_y_val = None
    combined_idx = None

    for arch_name, config in MODEL_CONFIGS.items():
        versions = [
            (arch_name + "_Keras", os.path.join(TRAINED_NETWORKS_DIR, config["keras"]), "keras"),
            (arch_name + "_16bit", os.path.join(TRAINED_NETWORKS_DIR, config["tflite_16"]), "tflite"),
            (arch_name + "_8bit", os.path.join(TRAINED_NETWORKS_DIR, config["tflite_8"]), "tflite")
        ]
        
        for full_net_name, model_path, m_type in versions:
            if not os.path.exists(model_path):
                print(f"[SKIP] Plik modelu nie istnieje: {model_path}")
                continue

            print(f"Walidacja: {full_net_name}...")
            report_dict[full_net_name] = {}
            
            all_y_v, all_y_p, all_idx = [], [], []
            current_offset = 0
            
            model_obj = None
            if m_type == "keras":
                try:
                    model_obj = keras.models.load_model(model_path, compile=False)
                except Exception as e:
                    print(f"[ERROR] Nie można wczytać modelu {full_net_name}: {e}")
                    continue

            for file_name in SELECTED_FILES_VALIDATION:
                f_path = os.path.join(DATA_DIR, file_name)
                if not os.path.exists(f_path):
                    continue
                
                X, y_scaled, idx = get_windows_single_file(f_path, scaler_x, scaler_y, config["window"], offset=current_offset)
                
                if len(X) == 0:
                    continue

                if m_type == "keras":
                    y_p_scaled = model_obj.predict(X, verbose=0).flatten()
                else:
                    y_p_scaled = predict_with_tflite(model_path, X)
                
                y_km = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
                y_p_km = scaler_y.inverse_transform(y_p_scaled.reshape(-1, 1)).flatten()
                
                f_base = file_name.replace(".csv", "")
                img_name = f"{f_base}_{full_net_name}.png"
                plot_validation_results(y_km, y_p_km, idx, os.path.join(VALIDATION_OUTPUT_DIR, img_name), 
                                        f"Validation: {full_net_name} | File: {file_name}")
                
                error_arr = np.abs(y_km - y_p_km)
                warmup_size = min(WARMUP_WINDOW, len(error_arr))
                
                max_err_warmup = np.max(error_arr[:warmup_size]) if warmup_size > 0 else 0.0
                max_err_after = np.max(error_arr[warmup_size:]) if len(error_arr) > warmup_size else 0.0
                
                report_dict[full_net_name][file_name] = {
                    "warmup_err": max_err_warmup,
                    "after_err": max_err_after
                }
                
                all_y_v.extend(y_km)
                all_y_p.extend(y_p_km)
                all_idx.extend(idx)
                current_offset += (len(y_km) * config["window"])

            if all_y_v:
                plot_validation_results(np.array(all_y_v), np.array(all_y_p), np.array(all_idx), 
                                        os.path.join(VALIDATION_OUTPUT_DIR, f"GLOBAL_VALIDATION_{full_net_name}.png"),
                                        f"Global Validation: {full_net_name} (All Files)")
                
                # --- ZAPISYWANIE WYNIKÓW DLA MODELI KERAS DO WYKRESU ZBIORCZEGO ---
                if m_type == "keras":
                    combined_keras_preds[arch_name] = np.array(all_y_p)
                    # Wartości rzeczywiste i indeksy są zawsze takie same dla każdego pliku, 
                    # więc zapisujemy je tylko raz.
                    if combined_y_val is None:
                        combined_y_val = np.array(all_y_v)
                        combined_idx = np.array(all_idx)
            
            if model_obj:
                keras.backend.clear_session()

    # --- GENEROWANIE WYKRESU ZBIORCZEGO ---
    if combined_keras_preds and combined_y_val is not None:
        print("\nGenerowanie wykresu zbiorczego dla modeli KERAS...")
        combined_plot_path = os.path.join(VALIDATION_OUTPUT_DIR, "COMBINED_GLOBAL_KERAS.png")
        plot_combined_validation_results(
            combined_y_val, 
            combined_keras_preds, 
            combined_idx, 
            combined_plot_path, 
            "Range Estimation Validation (KERAS)"
        )

    # 1. Generowanie raportu tekstowego ze szczegółami plików
    report_path = os.path.join(VALIDATION_OUTPUT_DIR, "validation_report_range.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("=== RANGE VALIDATION REPORT (BY NETWORK) ===\n\n")
        f.write(f"WARM-UP WINDOW: {WARMUP_WINDOW} points\n\n")
        for net, files in report_dict.items():
            f.write(f"MODEL: {net.upper()}\n")
            f.write("-" * 80 + "\n")
            for f_name, errs in files.items():
                f.write(f"  - {f_name:50}\n")
                f.write(f"      Max Error (Warm-up): {errs['warmup_err']:.2f} km\n")
                f.write(f"      Max Error (After):   {errs['after_err']:.2f} km\n")
            f.write("\n")

    # 2. Generowanie pliku table.txt
    table_path = os.path.join(VALIDATION_OUTPUT_DIR, "table.txt")
    with open(table_path, "w", encoding='utf-8') as f:
        f.write("=== ZBIORCZA TABELA BŁĘDÓW (RANGE) ===\n")
        f.write(f"Warm-up window = {WARMUP_WINDOW} próbek\n\n")
        
        header = f"{'Sieć / Model':<25} | {'Max Błąd (Warm-up) [km]':<25} | {'Max Błąd (Po Warm-up) [km]':<25}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for net, files in report_dict.items():
            if not files:
                continue
                
            global_max_warmup = max(errs['warmup_err'] for errs in files.values())
            global_max_after = max(errs['after_err'] for errs in files.values())
            
            f.write(f"{net:<25} | {global_max_warmup:<25.2f} | {global_max_after:<25.2f}\n")

    print(f"\n[DONE] Wyniki walidacji zasięgu zapisano w: {VALIDATION_OUTPUT_DIR}")
    print(f"[DONE] Tabela z podsumowaniem (table.txt) jest gotowa.")

if __name__ == "__main__":
    main()