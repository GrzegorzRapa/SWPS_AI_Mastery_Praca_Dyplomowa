#Driving Cycle Chooper
#This program processes existing fleet driving cycles, firstly by filtering driving cycle for paticular vehcle and then chopping it when speed crosses zero.
#source 'VED_' files must be moved from the 'processed' folder to the main program folder.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import shutil
import glob
import sys

# ================= CONFIGURATION =================
# 0 for all vehicles, or a list like [10, 531] to process specific IDs
TARGET_VEHICLES = 0

# Trip filtering
LONG_TRIP = 1  # 1 to enable filtering by duration, 0 to process all
MIN_LENGTH_S = 2400  # Minimum cycle duration in seconds (e.g., 60s)

ELEV_SAMPLING_M = 20    # Sample elevation every 20 meters to avoid ban at open-elevation
API_BATCH_SIZE = 100    #hard limit of open-elevation WEBSITE.
PROCESSED_DIR = 'processed'


# ================================================

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two GPS points."""
    R = 6371000
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + \
        np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_elevations_from_api(locations):
    """Fetches elevation data from Open-Elevation API."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    try:
        response = requests.post(url, json={"locations": locations}, timeout=40)
        if response.status_code == 429:
            print("\n[CRITICAL] API Quota exceeded.")
            sys.exit(1)
        if response.status_code == 200:
            return [r['elevation'] for r in response.json().get('results', [])]
        return None
    except Exception as e:
        print(f"\n[CONNECTION ERROR] {e}")
        return None


def generate_dual_plot(df, name):
    """Generates subplots for Speed and Elevation with Time in SECONDS."""
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    time_s = df['Timestamp(ms)'] / 1000

    # Plot 1: Vehicle Speed
    ax1.plot(time_s, df['Vehicle Speed[km/h]'], color='tab:blue', linewidth=1.5, label='Speed')
    ax1.set_ylabel('Vehicle Speed [km/h]', fontsize=10)
    ax1.set_title(f"Trip Profile: {name}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Plot 2: Elevation
    ax2.plot(time_s, df['Elevation [m]'], color='tab:red', linewidth=2, label='Elevation')
    ax2.set_ylabel('Elevation [m]', fontsize=10)
    ax2.set_xlabel('Time [s]', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close(fig)


def process_cycle(cycle_df, v_id, trip_id):
    """Processes a single trip: distance, sampling, and interpolation."""
    total_rows = len(cycle_df)

    # Calculate duration safely
    duration_s = (cycle_df['Timestamp(ms)'].max() - cycle_df['Timestamp(ms)'].min()) / 1000

    unique_name = f"Veh_{int(v_id)}_Trip_{int(trip_id)}"

    # Check long_trip condition
    if LONG_TRIP == 1 and duration_s < MIN_LENGTH_S:
        print(f"    -> Skipped: {unique_name} (Length: {duration_s:.1f}s < {MIN_LENGTH_S}s)")
        return

    print(f"    -> Processing: {unique_name} ({total_rows} rows, {duration_s:.1f}s)")

    # 1. Distance Calculation
    lats = cycle_df['Latitude[deg]'].values
    lons = cycle_df['Longitude[deg]'].values
    cum_dists = [0.0]
    for i in range(1, total_rows):
        d = calculate_haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        cum_dists.append(cum_dists[-1] + d)
    cycle_df['cum_dist'] = cum_dists

    # 2. Elevation Sampling (every 20m)
    sampled_indices = [0]
    last_d = 0
    for i in range(1, total_rows):
        if cum_dists[i] - last_d >= ELEV_SAMPLING_M:
            sampled_indices.append(i)
            last_d = cum_dists[i]
    if (total_rows - 1) not in sampled_indices:
        sampled_indices.append(total_rows - 1)

    # 3. Fetch Elevation
    sampled_locs = [{"latitude": lats[idx], "longitude": lons[idx]} for idx in sampled_indices]
    print(f"       API: Requesting {len(sampled_locs)} control points...")

    elev_results = []
    for start in range(0, len(sampled_locs), API_BATCH_SIZE):
        batch = sampled_locs[start: start + API_BATCH_SIZE]
        res = get_elevations_from_api(batch)
        if res:
            elev_results.extend(res)
        else:
            elev_results.extend([0.0] * len(batch))

    # 4. Interpolation
    cycle_df['Elevation [m]'] = np.nan
    for i, idx in enumerate(sampled_indices):
        cycle_df.iloc[idx, cycle_df.columns.get_loc('Elevation [m]')] = elev_results[i]

    cycle_df['Elevation [m]'] = cycle_df['Elevation [m]'].interpolate(method='linear')

    # 5. Save Results
    out_df = cycle_df[['Timestamp(ms)', 'Vehicle Speed[km/h]', 'Elevation [m]']]
    out_df.to_csv(f"{unique_name}.csv", index=False, sep=';')
    generate_dual_plot(cycle_df, unique_name)
    print(f"       [SUCCESS] Saved: {unique_name}.csv and .png")


def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    files = glob.glob("VED_*.csv")
    if not files:
        print("No files starting with 'VED_' found.")
        return

    for f_path in files:
        print(f"\n[FILE] Opening: {f_path}")
        try:
            df = pd.read_csv(f_path, sep=None, engine='python')
            df.columns = [c.strip() for c in df.columns]

            # Ensure the 'Trip' column exists
            if 'Trip' not in df.columns:
                print(f"  [ERROR] Column 'Trip' not found in {f_path}. Skipping file.")
                continue

            # Filter Target Vehicles if specified
            if TARGET_VEHICLES != 0:
                df = df[df['VehId'].isin(TARGET_VEHICLES)]
                if df.empty:
                    print(f"  -> No data for target vehicles in this file.")
                    continue

            # Group safely by Vehicle ID and Trip ID
            grouped = df.groupby(['VehId', 'Trip'])
            print(f"  -> Found {len(grouped)} total cycles in file.")

            for (v_id, trip_id), cycle_df in grouped:
                # Sort by time within the trip to ensure calculation is strictly sequential
                cycle_df = cycle_df.sort_values('Timestamp(ms)').reset_index(drop=True)

                # Process the cycle
                process_cycle(cycle_df, v_id, trip_id)

            # Move source file to processed directory
            shutil.move(f_path, os.path.join(PROCESSED_DIR, os.path.basename(f_path)))
            print(f"[STATUS] Source file moved to /{PROCESSED_DIR}")

        except Exception as e:
            print(f"  [ERROR] Failed to process {f_path}: {e}")


if __name__ == "__main__":
    main()