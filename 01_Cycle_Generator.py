import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SOURCE_DIR = 'source_data'
OUTPUT_DIR = 'generated_cycles'
T_DRIVE_S = 28800
SAMPLE_TIME_MS = 100
BLEND_WINDOW_S = 120
BLEND_SAMPLES = int((BLEND_WINDOW_S * 1000) / SAMPLE_TIME_MS)
N_CLUSTERS = 3
REQUIRED_MIN_HIGH_SPEED = 100.0
STRETCH_ACCEL_LIMIT = 2.5
TARGET_PROPORTIONS = {0: 0.1, 1: 0.89, 2: 0.01}


def close_to_zero(df, accel_limit=1.0):
    """Ensures fragment starts and ends at 0 km/h by adding ramps."""
    dt = SAMPLE_TIME_MS / 1000.0
    v_start = df['Vehicle Speed[km/h]'].iloc[0] / 3.6
    v_end = df['Vehicle Speed[km/h]'].iloc[-1] / 3.6

    if v_start > 0.05:
        t_ramp = np.arange(0, v_start / accel_limit, dt)
        v_ramp = accel_limit * t_ramp
        ramp_df = pd.DataFrame({
            'Timestamp(ms)': (t_ramp * 1000).astype(int),
            'Vehicle Speed[km/h]': (v_ramp * 3.6),
            'Elevation [m]': [df['Elevation [m]'].iloc[0]] * len(t_ramp)
        })
        df = pd.concat([ramp_df, df], ignore_index=True)

    if v_end > 0.05:
        t_ramp = np.arange(0, v_end / accel_limit, dt)
        v_ramp = np.clip(v_end - accel_limit * t_ramp, 0, None)
        last_t = df['Timestamp(ms)'].iloc[-1]
        ramp_df = pd.DataFrame({
            'Timestamp(ms)': (last_t + SAMPLE_TIME_MS + t_ramp * 1000).astype(int),
            'Vehicle Speed[km/h]': (v_ramp * 3.6),
            'Elevation [m]': [df['Elevation [m]'].iloc[-1]] * len(t_ramp)
        })
        df = pd.concat([df, ramp_df], ignore_index=True)
    return df


def extract_fragments_from_file(filepath):
    """Interpolates data and extracts microtrips between v=0 points."""
    try:
        df = pd.read_csv(filepath, sep=';')
        if 'Vehicle Speed[km/h]' not in df.columns: return []

        t_min, t_max = df['Timestamp(ms)'].min(), df['Timestamp(ms)'].max()
        new_t = np.arange(t_min, t_max + SAMPLE_TIME_MS, SAMPLE_TIME_MS)
        df_interp = pd.DataFrame({'Timestamp(ms)': new_t})
        df_interp['Vehicle Speed[km/h]'] = np.interp(new_t, df['Timestamp(ms)'], df['Vehicle Speed[km/h]']).clip(0)
        df_interp['Elevation [m]'] = np.interp(new_t, df['Timestamp(ms)'], df['Elevation [m]'])

        df_interp = close_to_zero(df_interp)
        speed = df_interp['Vehicle Speed[km/h]'].values
        zero_indices = np.where(speed < 0.1)[0]

        frags = []
        for i in range(len(zero_indices) - 1):
            s, e = zero_indices[i], zero_indices[i + 1]
            if e - s > 10:
                frag = df_interp.iloc[s:e + 1].copy()
                if frag['Vehicle Speed[km/h]'].max() > 1.0:
                    frags.append(frag)
        return frags
    except Exception:
        return []


def kinematic_solver(s_grid, v_target, e_s, a_max, d_max, ds=0.5):
    """Calculates v(t) profile from v(s) target with acceleration constraints."""
    v_res = np.zeros_like(s_grid)
    for i in range(1, len(s_grid)):
        v_res[i] = min(v_target[i], np.sqrt(v_res[i - 1] ** 2 + 2 * a_max * ds))
    for i in range(len(s_grid) - 2, -1, -1):
        v_res[i] = min(v_res[i], np.sqrt(v_res[i + 1] ** 2 + 2 * d_max * ds))

    t_new = [0.0]
    for i in range(1, len(s_grid)):
        v_avg = max(0.1, (v_res[i] + v_res[i - 1]) / 2.0)
        t_new.append(t_new[-1] + ds / v_avg)

    t_uniform = np.arange(0, t_new[-1], 0.1)
    return pd.DataFrame({
        'Timestamp(ms)': (t_uniform * 1000).astype(int),
        'Vehicle Speed[km/h]': (np.interp(t_uniform, t_new, v_res) * 3.6),
        'Elevation [m]': np.interp(t_uniform, t_new, e_s)
    })


def kinematic_stretch(frag_df, target_v):
    """Stretches fragments to reach highway speed targets physically."""
    v = frag_df['Vehicle Speed[km/h]'].values / 3.6
    if v.max() * 3.6 >= target_v: return frag_df
    dt = 0.1
    dist = np.cumsum(v * dt)
    ds = 0.5
    s_grid = np.arange(0, dist[-1], ds)
    v_s = np.interp(s_grid, dist, v)
    e_s = np.interp(s_grid, dist, frag_df['Elevation [m]'].values)
    v_target = v_s * (target_v / 3.6 / max(0.1, v_s.max()))
    return kinematic_solver(s_grid, v_target, e_s, STRETCH_ACCEL_LIMIT, STRETCH_ACCEL_LIMIT, ds)


def derive_kinematic_profile(base_df, style):
    """Generates eco and aggressive versions based on the base profile."""
    configs = {
        'eco': {'a': 1.0, 'd': 1.0, 'v_m': 0.85},
        'aggressive': {'a': 3.0, 'd': 4.0, 'v_m': 1.20}
    }
    c = configs[style]
    speed = base_df['Vehicle Speed[km/h]'].values / 3.6
    is_moving = speed > 0.05
    diff = np.diff(is_moving.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    new_frags = []
    last_idx = 0
    for s, e in zip(starts, ends):
        if s > last_idx:
            new_frags.append(base_df.iloc[last_idx:s].copy())
        seg = base_df.iloc[s:e].copy()
        v_seg = seg['Vehicle Speed[km/h]'].values / 3.6
        dist = np.cumsum(v_seg * 0.1)
        if dist[-1] < 0.5:
            new_frags.append(seg)
        else:
            ds = 0.5
            s_grid = np.arange(0, dist[-1], ds)
            v_target = np.interp(s_grid, dist, v_seg) * c['v_m']
            e_s = np.interp(s_grid, dist, seg['Elevation [m]'].values)
            new_frags.append(kinematic_solver(s_grid, v_target, e_s, c['a'], c['d'], ds))
        last_idx = e

    if last_idx < len(base_df):
        new_frags.append(base_df.iloc[last_idx:].copy())

    curr_t = 0
    for f in new_frags:
        f['Timestamp(ms)'] = np.arange(curr_t, curr_t + len(f) * 100, 100)
        curr_t = f['Timestamp(ms)'].iloc[-1] + 100
    return pd.concat(new_frags, ignore_index=True)


def blend_elevations(frag_list):
    """Sigmoidal blending for elevation between joined fragments."""
    for i in range(len(frag_list) - 1):
        curr_f, next_f = frag_list[i], frag_list[i + 1]
        w_b, w_f = min(BLEND_SAMPLES, len(curr_f)), min(BLEND_SAMPLES, len(next_f))
        e_s, e_e = curr_f['Elevation [m]'].iloc[-w_b], next_f['Elevation [m]'].iloc[w_f - 1]
        weights = 1 / (1 + np.exp(-np.linspace(-6, 6, w_b + w_f)))
        blended = e_s + weights * (e_e - e_s)
        curr_f.iloc[-w_b:, curr_f.columns.get_loc('Elevation [m]')] = blended[:w_b]
        next_f.iloc[:w_f, next_f.columns.get_loc('Elevation [m]')] = blended[w_b:]
    return frag_list


def calculate_distance_km(df):
    return (df['Vehicle Speed[km/h]'].sum() * 0.1) / 3600


def format_final_df(df):
    """Rounds velocity and elevation to 2 decimal places as requested."""
    df['Vehicle Speed[km/h]'] = df['Vehicle Speed[km/h]'].round(2)
    df['Elevation [m]'] = df['Elevation [m]'].round(2)
    return df


def generate_cycle():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(SOURCE_DIR, 'Veh_*.csv'))
    if not files: return

    all_frags = []
    for f in files: all_frags.extend(extract_fragments_from_file(f))

    feats = np.array([[np.mean(f['Vehicle Speed[km/h]']), f['Vehicle Speed[km/h]'].max()] for f in all_frags])
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit_predict(
        StandardScaler().fit_transform(feats))
    c_speeds = [np.mean(feats[labels == i, 0]) for i in range(N_CLUSTERS)]
    mapping = {old: new for new, old in enumerate(np.argsort(c_speeds))}
    clustered = {i: [] for i in range(N_CLUSTERS)}
    for i, f in enumerate(all_frags): clustered[mapping[labels[i]]].append(f)

    target_samples = int(T_DRIVE_S * 10)
    selected_frags = []
    curr_samples = 0
    highway_idx = N_CLUSTERS - 1

    while curr_samples < target_samples:
        c = random.choices(range(N_CLUSTERS), weights=[TARGET_PROPORTIONS[i] for i in range(N_CLUSTERS)])[0]
        frag = random.choice(clustered[c]).copy()
        if c == highway_idx and frag['Vehicle Speed[km/h]'].max() < REQUIRED_MIN_HIGH_SPEED:
            frag = kinematic_stretch(frag, REQUIRED_MIN_HIGH_SPEED)
        selected_frags.append(frag)
        curr_samples += len(frag)

    selected_frags = blend_elevations(selected_frags)
    curr_t = 0
    for f in selected_frags:
        f['Timestamp(ms)'] = np.arange(curr_t, curr_t + len(f) * 100, 100)
        curr_t = f['Timestamp(ms)'].iloc[-1] + 100

    df_normal = format_final_df(pd.concat(selected_frags, ignore_index=True))
    df_eco = format_final_df(derive_kinematic_profile(df_normal, 'eco'))
    df_aggr = format_final_df(derive_kinematic_profile(df_normal, 'aggressive'))

    # Get sequence number
    existing = glob.glob(os.path.join(OUTPUT_DIR, 'generated_cycle_*_NORMAL.csv'))
    xxx = max([int(f.split('_')[-2]) for f in existing if f.split('_')[-2].isdigit()] + [0]) + 1
    base_name = f"generated_cycle_{xxx:03d}"

    # Save CSVs with 2-decimal precision
    df_eco.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_ECO.csv"), sep=';', index=False)
    df_normal.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_NORMAL.csv"), sep=';', index=False)
    df_aggr.to_csv(os.path.join(OUTPUT_DIR, f"{base_name}_AGGRESIVE.csv"), sep=';', index=False)

    print(f"CYCLE {xxx:03d} SUMMARY:")
    for label, df in [('ECO', df_eco), ('NORMAL', df_normal), ('AGGRESIVE', df_aggr)]:
        print(f"{label}: Dist = {calculate_distance_km(df):.3f} km, Time = {len(df) / 10:.1f} s")

    # Plotting with individual timebases
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for i, (label, df) in enumerate([('ECO', df_eco), ('NORMAL', df_normal), ('AGGRESIVE', df_aggr)]):
        t_s = df['Timestamp(ms)'] / 1000.0
        axes[i, 0].plot(t_s, df['Vehicle Speed[km/h]'], color='blue')
        axes[i, 0].set_title(f"{label} Speed (Max: {df['Vehicle Speed[km/h]'].max()} km/h)")
        axes[i, 0].grid(True)
        axes[i, 1].plot(t_s, df['Elevation [m]'], color='green')
        axes[i, 1].set_title(f"{label} Elevation")
        axes[i, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}.png"))


if __name__ == "__main__":
    generate_cycle()