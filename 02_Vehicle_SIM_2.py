import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator
import os
import json
import math

# ==============================================================================
# CONFIGURATION DATA
# ==============================================================================

# Training - from 100% to min cell voltage
# temperatures: -20, 0, 25

INPUT_DIR = 'generated_cycles'
OUTPUT_DIR = 'driving_cycle_results' 
OUTPUT_PREFIX = 'additonal_test_cycle_soc50_0_0degC_'
SIMULATION_TIME_STEP_S = 0.1
SMOOTHING_WINDOW_S = 30 

ENABLE_200MS_RESAMPLING = False  # Przełącznik do uśredniania z 100ms do 200ms

PROCESS_ALL_FILES = False
SELECTED_FILES = ['generated_cycle_001_NORMAL.csv',
                  'generated_cycle_003_AGGRESIVE.csv',
                  'generated_cycle_011_ECO.csv',
                  'generated_cycle_007_ECO.csv'
                 ]

# PACK CONFIGURATION
PACK_Ns = 165   
PACK_Np = 20

# Pojedyncze zmienne początkowe dla całego pakietu
PACK_INITIAL_TEMP = 2.0 
PACK_INITIAL_SOC = 0.50

OUTER_CELLS_RATIO = 0.4  
INNER_CELLS_RATIO = 0.6  
INNER_HEAT_TRANSFER_PENALTY = 0.15 

CONFIG_CELL_TYPE = "LGM50_NMC_5Ah"

MIN_CELL_VOLTAGE_V = 2.65 
MAX_CELL_VOLTAGE_V = 4.20  
MIN_SOC_CUTOFF_PCT = 3.00  

HEATING_MAX_TEMP_degC = 15.0    
HEATING_START_TEMP_degC = 18.0  
HEATING_STOP_TEMP_deg_C = 19.0  
HEATING_MAX_POWER_W = 3000.0 

COOLING_START_TEMP_deg_C = 25.0 
COOLING_MAX_TEMP_degC = 35.0 
COOLING_MAX_POWER_W = 7000.0 
COOLING_POWER_CONS_W = 2000.0 

MAX_BATT_TEMP_deg_C = 46.0

# CAR PARAMETERS:
car_mass_kg = 1950.0                 
car_mass_rotational_factor = 1.05
crr = 0.020
cd = 0.308                        
frontal_area_m2 = 2.4
tire_diameter_mm = 457.2

# CAR DRIVETRAIN PARAMETERS
auxiliary_power_W = 1000.0        
max_power_w = 100000.0            
max_regen_w = -35000.0            
gear_ratio = 9.0

# ENVIRONMENT:
air_density_kgm3 = 1.225          
gravity_ms2 = 9.81

# ==============================================================================
# INITIALIZATION & DATA LOADING
# ==============================================================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Load Efficiency Map 
try:
    eff_df = pd.read_csv('pmsm_100kW_efficiency_map.csv', sep=None, engine='python', encoding='utf-8-sig')
    eff_df.columns = eff_df.columns.str.strip()
    
    points = eff_df[['speed_rpm', 'torque_Nm']].values
    values = eff_df['efficiency_system'].values / 100.0 
    eff_interp = LinearNDInterpolator(points, values)
    eff_nearest = NearestNDInterpolator(points, values)

    def get_efficiency(rpm, torque):
        rpm_abs = abs(rpm)
        torque_abs = abs(torque)
        eff = eff_interp(rpm_abs, torque_abs)
        if np.isnan(eff):
            eff = eff_nearest(rpm_abs, torque_abs)
        return float(eff)
except FileNotFoundError:
    print("[FATAL] pmsm_100kW_efficiency_map.csv not found.")
    exit(1)

# 2. Load Battery Cell Parameters
try:
    with open('battery_cell_library.json', 'r') as f:
        cell_lib = json.load(f)
except FileNotFoundError:
    print("[FATAL] battery_cell_library.json not found.")
    exit(1)

cell_data = next((item for item in cell_lib if item["id"] == CONFIG_CELL_TYPE), None)
if not cell_data:
    print(f"[FATAL] Cell {CONFIG_CELL_TYPE} not found in library.")
    exit(1)

temps = sorted(cell_data["temperatures_C"])
socs = cell_data["soc_breakpoints"]

def build_interpolator(param_table):
    grid = np.zeros((len(temps), len(socs)))
    for i, t in enumerate(temps):
        t_key = str(t)
        if t_key in param_table:
            grid[i, :] = param_table[t_key]
        elif "all" in param_table:
            grid[i, :] = param_table["all"]
    return RegularGridInterpolator((temps, socs), grid, bounds_error=False, fill_value=None)

ocv_interp = build_interpolator(cell_data["ocv_table"])
r0_interp = build_interpolator(cell_data["R0_table"])
r1_interp = build_interpolator(cell_data["R1_table"])
c1_interp = build_interpolator(cell_data["C1_table"])
r2_interp = build_interpolator(cell_data["R2_table"])
c2_interp = build_interpolator(cell_data["C2_table"])

thermal_mass_per_cell = cell_data["thermal_mass_J_per_K"]
heat_transfer_per_cell = cell_data["heat_transfer_W_per_K"]
cell_capacity_Ah = cell_data["capacity_Ah"]
cell_capacity_As = cell_capacity_Ah * 3600.0

total_battery_capacity_kWh = (PACK_Ns * PACK_Np * cell_capacity_Ah * cell_data["nominal_voltage_V"]) / 1000.0
print(f"Total Battery Capacity: {total_battery_capacity_kWh:.2f} kWh\n")

# ==============================================================================
# SIMULATION CORE
# ==============================================================================

files_to_process = os.listdir(INPUT_DIR) if PROCESS_ALL_FILES else SELECTED_FILES

for file_name in files_to_process:
    file_path = os.path.join(INPUT_DIR, file_name)
    if not os.path.exists(file_path):
        continue

    print(f"Processing {file_name}...")
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[ERROR] Could not read {file_name}: {e}")
        continue
        
    # RESAMPLING: Uśrednianie do 200ms
    if ENABLE_200MS_RESAMPLING:
        df = df.groupby(df.index // 2).mean()
        current_dt_s = 0.2
    else:
        current_dt_s = SIMULATION_TIME_STEP_S

    n_steps = len(df)
    time_ms = df['Timestamp(ms)'].values
    speed_kmh_requested = df['Vehicle Speed[km/h]'].values
    elevation_m = df['Elevation [m]'].values
    
    # State arrays
    actual_speed_kmh = np.zeros(n_steps)
    actual_distance_m = 0.0
    trace_distance_m = 0.0
    distance_deficit_m = 0.0
    
    batt_current_A = np.zeros(n_steps)
    batt_voltage_V = np.zeros(n_steps)
    batt_power_W = np.zeros(n_steps)
    
    pack_soc = np.zeros(n_steps)
    pack_temp = np.zeros(n_steps)
    
    # 2RC specific states for the pack
    v_c1 = 0.0
    v_c2 = 0.0
    
    pack_soc[0] = PACK_INITIAL_SOC
    pack_temp[0] = PACK_INITIAL_TEMP
    
    heating_active = False
    last_soc_update = int(PACK_INITIAL_SOC * 100)
    
    results = []

    for i in range(n_steps - 1):
        dt = current_dt_s
        current_soc = pack_soc[i]
        current_temp = pack_temp[i]
        
        current_soc_pct = current_soc * 100.0
        dist_km_current = actual_distance_m / 1000.0
        
        # Stop conditions
        if current_soc_pct <= MIN_SOC_CUTOFF_PCT:
            print(f"Simulation stopped: Minimum SoC cutoff reached ({current_soc_pct:.1f}%). Final distance: {dist_km_current:.2f} km.")
            break
        if current_temp >= MAX_BATT_TEMP_deg_C:
            print(f"Simulation stopped: Maximum temperature reached ({current_temp:.1f} °C). Final distance: {dist_km_current:.2f} km.")
            break
            
        if i > 0:
            min_cell_v_actual = batt_voltage_V[i-1] / PACK_Ns
            if min_cell_v_actual <= MIN_CELL_VOLTAGE_V:
                print(f"Simulation stopped: Minimum cell voltage cutoff reached ({min_cell_v_actual:.2f} V). Final distance: {dist_km_current:.2f} km.")
                break
            
        if int(current_soc_pct) < last_soc_update:
            print(f"Progress: Battery SoC at {current_soc_pct:.1f}%, Driven distance: {dist_km_current:.2f} km")
            last_soc_update = int(current_soc_pct)

        # 1. Physics & Target Dynamics
        v_actual_ms = actual_speed_kmh[i] / 3.6
        v_trace_ms = speed_kmh_requested[i] / 3.6
        v_next_trace_ms = speed_kmh_requested[i+1] / 3.6
        
        target_v_next = v_next_trace_ms + (distance_deficit_m / SMOOTHING_WINDOW_S)
        target_accel = (target_v_next - v_actual_ms) / dt
        
        v_avg_ms = (v_actual_ms + target_v_next) / 2.0
        
        # --- ZMIANA: Obliczenia nachylenia w procentach ---
        grade_rad = 0.0
        elevation_pct = 0.0  # Domyślna wartość
        
        if i < n_steps - 1 and v_avg_ms > 0:
            dx = v_avg_ms * dt
            dy = elevation_m[i+1] - elevation_m[i]
            if dx > 0: 
                grade_rad = math.atan2(dy, dx)
                elevation_pct = (dy / dx) * 100.0  # Obliczenie procentowego nachylenia drogi
        # --------------------------------------------------

        F_aero = 0.5 * air_density_kgm3 * cd * frontal_area_m2 * (v_avg_ms**2)
        F_roll = crr * car_mass_kg * gravity_ms2 * math.cos(grade_rad) if (v_avg_ms > 0 or target_accel > 0) else 0.0
        F_grade = car_mass_kg * gravity_ms2 * math.sin(grade_rad)
        F_accel = car_mass_kg * car_mass_rotational_factor * target_accel
        
        F_total = F_aero + F_roll + F_grade + F_accel
        P_mech_W = F_total * v_avg_ms
        
        wheel_rpm = v_avg_ms * 60 / (math.pi * tire_diameter_mm / 1000)
        motor_rpm = wheel_rpm * gear_ratio
        
        motor_torque_Nm = 0.0
        if motor_rpm > 1.0:
            motor_torque_Nm = P_mech_W / (motor_rpm * 2 * math.pi / 60)
        elif F_total > 0:
            motor_torque_Nm = F_total * (tire_diameter_mm / 2000) / gear_ratio
            
        # 2. Drivetrain Power Lookup
        eff = get_efficiency(motor_rpm, motor_torque_Nm)
        eff_drive = max(0.1, eff) 
        
        if P_mech_W >= 0:
            P_elec_drivetrain = P_mech_W / eff_drive 
        else:
            P_elec_drivetrain = P_mech_W * eff  
            
        P_elec_drivetrain = np.clip(P_elec_drivetrain, max_regen_w, max_power_w)

        # 3. Thermal Management Power
        p_cooling_w = 0.0
        p_cooling_removed_w = 0.0
        if current_temp >= COOLING_START_TEMP_deg_C:
            fraction = min(1.0, (current_temp - COOLING_START_TEMP_deg_C) / (COOLING_MAX_TEMP_degC - COOLING_START_TEMP_deg_C))
            p_cooling_removed_w = fraction * COOLING_MAX_POWER_W
            p_cooling_w = fraction * COOLING_POWER_CONS_W

        if current_temp < HEATING_START_TEMP_degC:
            heating_active = True
        if current_temp >= HEATING_STOP_TEMP_deg_C:
            heating_active = False
            
        p_heating_w = 0.0
        if heating_active:
            if current_temp <= HEATING_MAX_TEMP_degC:
                p_heating_w = HEATING_MAX_POWER_W
            else:
                fraction = 1.0 - (current_temp - HEATING_MAX_TEMP_degC) / (HEATING_START_TEMP_degC - HEATING_MAX_TEMP_degC)
                p_heating_w = fraction * HEATING_MAX_POWER_W
        
        P_batt_requested = P_elec_drivetrain + auxiliary_power_W + p_cooling_w + p_heating_w

        # 4. Rigorous Pack Battery Calculation
        pts = np.array([[current_temp, current_soc]])
        ocv_cell = ocv_interp(pts)[0]
        r0_cell = r0_interp(pts)[0]
        r1_cell = r1_interp(pts)[0]
        c1_cell = c1_interp(pts)[0]
        r2_cell = r2_interp(pts)[0]
        c2_cell = c2_interp(pts)[0]
            
        V_ocv_pack = PACK_Ns * (ocv_cell - v_c1 - v_c2)
        R0_pack = PACK_Ns * r0_cell
        
        a = PACK_Np * R0_pack
        b = -PACK_Np * V_ocv_pack
        c_quad = P_batt_requested
        
        disc_est = b**2 - 4*a*c_quad
        I_cell_est = (-b - math.sqrt(disc_est))/(2*a) if disc_est >= 0 else -b/(2*a)
        
        derated = False
        friction_brake_power = 0.0
        
        # Check Limits
        v_cell_pred = ocv_cell - v_c1 - v_c2 - I_cell_est * r0_cell
        
        if v_cell_pred > MAX_CELL_VOLTAGE_V and P_batt_requested < 0:
            I_cell_max = (ocv_cell - v_c1 - v_c2 - MAX_CELL_VOLTAGE_V) / r0_cell
            P_batt_allowed = (I_cell_max * PACK_Np) * (V_ocv_pack - I_cell_max * R0_pack)
            if P_batt_allowed > P_batt_requested: 
                friction_brake_power = abs(P_batt_requested - P_batt_allowed)
                P_batt_requested = P_batt_allowed
                derated = True
                
        elif v_cell_pred < MIN_CELL_VOLTAGE_V and P_batt_requested > 0:
            I_cell_max = (ocv_cell - v_c1 - v_c2 - MIN_CELL_VOLTAGE_V) / r0_cell
            P_batt_allowed = (I_cell_max * PACK_Np) * (V_ocv_pack - I_cell_max * R0_pack)
            if P_batt_allowed < P_batt_requested:
                P_batt_requested = P_batt_allowed
                print(f"[LIMIT] Power reduced due to MIN_CELL_VOLTAGE at {time_ms[i]} ms")
                derated = True
                
        # Final Exact Current calculation
        c_quad = P_batt_requested
        discriminant = b**2 - 4*a*c_quad
        if discriminant < 0:
            I_cell_actual = -b / (2*a) 
        else:
            I_cell_actual = (-b - math.sqrt(discriminant)) / (2*a)
            
        I_batt_actual = I_cell_actual * PACK_Np
        
        # 5. Update Pack (2RC & Thermal)
        v_c1 += dt * (I_cell_actual / c1_cell - v_c1 / (r1_cell * c1_cell))
        v_c2 += dt * (I_cell_actual / c2_cell - v_c2 / (r2_cell * c2_cell))
        
        v_cell_actual = ocv_cell - I_cell_actual * r0_cell - v_c1 - v_c2
        pack_voltage = v_cell_actual * PACK_Ns
        
        pack_soc[i+1] = current_soc - (I_cell_actual * dt) / cell_capacity_As
        
        # Thermal Update
        p_heat_cell_w = (I_cell_actual**2 * r0_cell) + (v_c1**2 / r1_cell) + (v_c2**2 / r2_cell)
        total_cells_in_pack = PACK_Ns * PACK_Np
        q_gen_w = p_heat_cell_w * total_cells_in_pack
        
        effective_heat_transfer_outer = (total_cells_in_pack * OUTER_CELLS_RATIO) * heat_transfer_per_cell
        effective_heat_transfer_inner = (total_cells_in_pack * INNER_CELLS_RATIO) * (heat_transfer_per_cell * INNER_HEAT_TRANSFER_PENALTY)
        total_effective_heat_transfer = effective_heat_transfer_outer + effective_heat_transfer_inner
        
        q_env_w = total_effective_heat_transfer * (current_temp - 20.0) 
        
        thermal_mass_pack = thermal_mass_pack = thermal_mass_per_cell * total_cells_in_pack
        dT = (q_gen_w + p_heating_w - p_cooling_removed_w - q_env_w) / thermal_mass_pack * dt
        pack_temp[i+1] = current_temp + dT

        batt_voltage_V[i] = pack_voltage
        batt_current_A[i] = I_batt_actual
        batt_power_W[i] = pack_voltage * I_batt_actual
        
        # 6. Resolve Actual Kinematics
        actual_P_mech = batt_power_W[i] - auxiliary_power_W - p_cooling_w - p_heating_w
        if actual_P_mech >= 0:
            actual_P_mech *= max(0.1, eff)
        else:
            actual_P_mech = (actual_P_mech / eff) if eff > 0.1 else actual_P_mech
            actual_P_mech -= friction_brake_power 
            
        if v_avg_ms > 0.01:
            actual_F_total = actual_P_mech / v_avg_ms
        else:
            actual_F_total = F_total if actual_P_mech > 0 else 0.0
            
        actual_F_accel = actual_F_total - F_aero - F_roll - F_grade
        actual_accel = actual_F_accel / (car_mass_kg * car_mass_rotational_factor)
        
        v_next_actual_ms = v_actual_ms + actual_accel * dt
        v_next_actual_ms = max(0.0, v_next_actual_ms)
        
        actual_speed_kmh[i+1] = v_next_actual_ms * 3.6
        
        # Update Deficits for catch-up
        trace_distance_m += v_trace_ms * dt
        actual_distance_m += v_actual_ms * dt
        distance_deficit_m = trace_distance_m - actual_distance_m
        
        driven_distance_km = round(actual_distance_m / 1000.0, 2)
        
        # --- ZMIANA: Przebudowana lista zapisu pojedynczego wiersza ---
        row = [
            time_ms[i], 
            driven_distance_km, 
            actual_speed_kmh[i],     
            actual_accel,            # Nowa zmienna: Acceleration[m/s2]
            elevation_m[i],          # Wysokość
            elevation_pct,           # Nowa zmienna: Elevation[pct]
            batt_voltage_V[i], 
            batt_current_A[i], 
            batt_power_W[i], 
            current_soc * 100, 
            current_temp,
            speed_kmh_requested[i]   
        ]
        results.append(row)

    # --- ZMIANA: Nowa definicja kolumn przed konwersją do DataFrame ---
    cols = [
        'Timestamp(ms)', 
        'Driven_Distance[km]', 
        'Vehicle_Speed[km/h]', 
        'Acceleration[m/s2]',        # Dodane
        'Elevation[m]',              # Zmieniona nazwa
        'Elevation[pct]',            # Dodane
        'Battery_Voltage[V]', 
        'Battery_Current[A]', 
        'Battery_Power[W]', 
        'Battery SoC[%]', 
        'Battery_Temp[degC]', 
        'Requested_Vehicle_Speed[km/h]'
    ]
        
    out_df = pd.DataFrame(results, columns=cols)
    
    # --- ZMIANA: Obliczenie i wstawienie Remaining_Distance[km] ---
    # Odczytujemy ostatnią wartość przejechanego dystansu dla całej symulacji
    total_distance = out_df['Driven_Distance[km]'].iloc[-1]
    
    # Wstawiamy nową kolumnę na indeks 2 (czyli zaraz po 'Driven_Distance[km]')
    out_df.insert(2, 'Remaining_Distance[km]', total_distance - out_df['Driven_Distance[km]'])
    # --------------------------------------------------------------

    csv_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{file_name}")
    out_df.to_csv(csv_path, index=False, sep=';')
    
    # Plotting Output
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_s = out_df['Timestamp(ms)'] / 1000.0
    
    for ax in axes.flat:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
    # Plot 1: Rzeczywista Prędkość vs Oczekiwana
    ax1 = axes[0, 0]
    ax1.plot(time_s, out_df['Requested_Vehicle_Speed[km/h]'], 'k--', label='Requested Speed', alpha=0.6)
    ax1.plot(time_s, out_df['Vehicle_Speed[km/h]'], 'b-', label='Actual Speed')
    ax1.set_ylabel('Speed [km/h]')
    ax1.set_xlabel('Time [s]')
    ax1_2 = ax1.twinx()
    
    # Zaktualizowana nazwa kolumny w wykresie
    ax1_2.plot(time_s, out_df['Elevation[m]'], 'g-', label='Elevation')
    ax1_2.set_ylabel('Elevation [m]')
    ax1.legend(loc='upper left')
    ax1_2.legend(loc='upper right')
    ax1.set_title("Vehicle Speed & Elevation")
    
    # Plot 2: Napięcie pakietu i Prąd
    ax2 = axes[0, 1]
    ax2.plot(time_s, out_df['Battery_Voltage[V]'], 'b-', label='Voltage [V]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_xlabel('Time [s]')
    ax2_2 = ax2.twinx()
    ax2_2.plot(time_s, out_df['Battery_Current[A]'], 'r-', label='Current [A]')
    ax2_2.set_ylabel('Current [A]')
    ax2.set_title("Pack Voltage & Current")
    
    # Plot 3: Globalne SoC & Temperatura
    ax3 = axes[1, 0]
    ax3.plot(time_s, out_df['Battery SoC[%]'], 'g-', label='SoC [%]')
    ax3.set_ylabel('SoC [%]')
    ax3.set_xlabel('Time [s]')
    ax3_2 = ax3.twinx()
    ax3_2.plot(time_s, out_df['Battery_Temp[degC]'], 'r-', label='Temp [°C]')
    ax3_2.set_ylabel('Temp [°C]')
    ax3.set_title("Pack SoC & Temperature")
    
    # Plot 4: Dystans i SoC
    ax4 = axes[1, 1]
    ax4.plot(time_s, out_df['Battery SoC[%]'], 'g-', label='SoC [%]')
    ax4.set_ylabel('SoC [%]')
    ax4.set_xlabel('Time [s]')
    ax4_2 = ax4.twinx()
    ax4_2.plot(time_s, out_df['Driven_Distance[km]'], 'm-', label='Distance [km]')
    ax4_2.set_ylabel('Distance [km]')
    ax4.legend(loc='upper left')
    ax4_2.legend(loc='upper right')
    ax4.set_title("SoC vs Driven Distance")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{file_name.replace('.csv', '.png')}"))
    plt.close()
    
    print(f"Finished {file_name}. Results saved in {OUTPUT_DIR}.")