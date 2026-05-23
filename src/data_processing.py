import scipy.io
import os
import glob
import numpy as np
import pandas as pd
import pickle
import time

def process_all_data():
    folders = [
        "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post",
        "RW_Skewed_High_Room_Temp_DataSet_2Post",
        "RW_Skewed_High_40C_DataSet_2Post"
    ]
    
    base_dir = "/Users/akashsunilsomsetwar/Desktop/edi_4"
    output_dir = os.path.join(base_dir, "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    all_features = []
    lstm_sequences = []
    lstm_targets_soh = []
    lstm_targets_rul = []
    lstm_meta = []
    
    print("Starting Data Preprocessing and Feature Extraction...")
    
    for folder in folders:
        # Determine condition name
        if "Uniform" in folder:
            condition = "Uniform RT"
        elif "40C" in folder:
            condition = "Skewed High 40C"
        else:
            condition = "Skewed RT"
            
        path = os.path.join(base_dir, folder, "data/Matlab/*.mat")
        mat_files = sorted(glob.glob(path))
        
        for mf in mat_files:
            filename = os.path.basename(mf)
            battery_id = filename.split(".")[0]
            print(f"Processing {battery_id} ({condition})...")
            
            try:
                start_t = time.time()
                mat_data = scipy.io.loadmat(mf)
                step = mat_data['data'][0, 0]['step']
                n_steps = step.shape[1]
                
                # 1. Find indices of reference discharge steps
                ref_dis_indices = []
                for i in range(n_steps):
                    c = step[0, i]['comment']
                    if c.size > 0 and str(c[0]) == 'reference discharge':
                        ref_dis_indices.append(i)
                
                # 2. Extract capacities and clean aborted cycles
                raw_capacities = []
                valid_dis_indices = []
                
                for idx in ref_dis_indices:
                    v = step[0, idx]['voltage'][0]
                    curr = step[0, idx]['current'][0]
                    t = step[0, idx]['relativeTime'][0]
                    
                    if len(t) < 50 or len(v) < 50:
                        continue  # Skip aborted cycles with too few data points
                        
                    dt = np.diff(t)
                    avg_curr = (curr[:-1] + curr[1:]) / 2.0
                    cap = np.sum(dt * avg_curr) / 3600.0
                    
                    if cap < 0.5:
                        continue  # Skip aborted cycles with negligible capacity
                        
                    raw_capacities.append(cap)
                    valid_dis_indices.append(idx)
                
                n_cycles = len(raw_capacities)
                if n_cycles == 0:
                    print(f"  Warning: No valid reference discharge cycles for {battery_id}")
                    continue
                
                # Initial capacity as nominal capacity for SOH calculation
                nominal_capacity = raw_capacities[0]
                soh_pct = [c / nominal_capacity * 100.0 for c in raw_capacities]
                
                # 3. Determine EOL cycle and calculate RUL
                # EOL is when SOH crosses below 80%
                cross_80 = None
                for i, s in enumerate(soh_pct):
                    if s < 80.0:
                        cross_80 = i + 1
                        break
                
                if cross_80 is not None:
                    eol_cycle = cross_80
                else:
                    # Extrapolate using linear regression on the SOH curve
                    cycles_arr = np.arange(1, n_cycles + 1)
                    slope, intercept = np.polyfit(cycles_arr, soh_pct, 1)
                    if slope < 0:
                        eol_cycle = int((80.0 - intercept) / slope)
                        # Cap EOL cycle to twice the total cycles to prevent runaway values
                        eol_cycle = min(eol_cycle, n_cycles * 2)
                    else:
                        eol_cycle = n_cycles
                        
                ruls = [max(0, eol_cycle - (i + 1)) for i in range(n_cycles)]
                
                # 4. Extract detailed features per valid cycle
                battery_features = []
                
                for k, idx in enumerate(valid_dis_indices):
                    v = step[0, idx]['voltage'][0]
                    curr = step[0, idx]['current'][0]
                    t = step[0, idx]['relativeTime'][0]
                    temp = np.clip(step[0, idx]['temperature'][0], 15.0, 60.0)
                    date_str = str(step[0, idx]['date'][0])
                    
                    # Capacity & SOH
                    capacity = raw_capacities[k]
                    soh = soh_pct[k]
                    rul = ruls[k]
                    
                    # Internal Resistance (Ri)
                    ri = 0.0
                    if idx > 0:
                        prev_v = step[0, idx-1]['voltage'][0]
                        prev_curr = step[0, idx-1]['current'][0]
                        if len(prev_v) > 0 and len(prev_curr) > 0:
                            v_prev_end = prev_v[-1]
                            curr_prev_end = prev_curr[-1]
                            
                            v_start = v[0]
                            curr_start = curr[0]
                            
                            dV = v_prev_end - v_start
                            dI = curr_start - curr_prev_end
                            
                            if abs(dI) > 0.1:
                                ri = dV / dI
                    
                    # If Ri is anomalous or 0, carry over previous or estimate from adjacent steps
                    if ri <= 0.01 or ri > 0.5:
                        if len(battery_features) > 0:
                            ri = battery_features[-1]['ri']
                        else:
                            ri = 0.08  # Default baseline for early cycles
                            
                    # Peak temperature
                    peak_temp = np.max(temp) if len(temp) > 0 else 25.0
                    
                    # Voltage drop
                    voltage_drop = v[0] - v[-1] if len(v) > 0 else 0.0
                    
                    # Duration
                    duration = t[-1] - t[0] if len(t) > 0 else 0.0
                    
                    # Preceding Random Walk steps features
                    prev_idx = valid_dis_indices[k-1] if k > 0 else 0
                    rw_temps = []
                    for i in range(prev_idx + 1, idx):
                        comm = step[0, i]['comment']
                        if comm.size > 0 and 'random walk' in str(comm[0]):
                            t_val = step[0, i]['temperature'][0]
                            if len(t_val) > 0:
                                rw_temps.extend(np.clip(t_val, 15.0, 60.0))
                                
                    if len(rw_temps) > 0:
                        mean_rw_temp = np.mean(rw_temps)
                        max_rw_temp = np.max(rw_temps)
                    else:
                        if len(battery_features) > 0:
                            mean_rw_temp = battery_features[-1]['mean_rw_temp']
                            max_rw_temp = battery_features[-1]['max_rw_temp']
                        else:
                            mean_rw_temp = 25.0
                            max_rw_temp = 25.0
                    
                    # Save features
                    feat_dict = {
                        'battery_id': battery_id,
                        'condition': condition,
                        'cycle': k + 1,
                        'date': date_str,
                        'capacity': capacity,
                        'soh': soh,
                        'rul': rul,
                        'ri': ri,
                        'peak_temp': peak_temp,
                        'voltage_drop': voltage_drop,
                        'duration': duration,
                        'mean_rw_temp': mean_rw_temp,
                        'max_rw_temp': max_rw_temp
                    }
                    battery_features.append(feat_dict)
                    all_features.append(feat_dict)
                    
                    # 5. Extract and downsample raw cycle curves for LSTM
                    # Sequence input is (voltage, current, temperature, elapsed_time, cumulative_capacity, cycle_normalized, ri, mean_rw_temp, max_rw_temp)
                    # We interpolate to exactly 100 points
                    t_new = np.linspace(t[0], t[-1], 100)
                    v_new = np.interp(t_new, t, v)
                    curr_new = np.interp(t_new, t, curr)
                    temp_new = np.interp(t_new, t, temp)
                    
                    elapsed_time = t_new - t_new[0]
                    dt_new = np.diff(t_new)
                    dt_new = np.insert(dt_new, 0, 0.0)
                    cumulative_capacity = np.cumsum(dt_new * abs(curr_new)) / 3600.0
                    cycle_normalized = np.full(100, (k + 1) / 100.0)
                    ri_channel = np.full(100, ri)
                    mean_rw_channel = np.full(100, mean_rw_temp)
                    max_rw_channel = np.full(100, max_rw_temp)
                    
                    seq_matrix = np.stack([
                        v_new, curr_new, temp_new, 
                        elapsed_time, cumulative_capacity, 
                        cycle_normalized, ri_channel, 
                        mean_rw_channel, max_rw_channel
                    ], axis=1)  # shape (100, 9)
                    lstm_sequences.append(seq_matrix)
                    lstm_targets_soh.append(soh)
                    lstm_targets_rul.append(rul)
                    lstm_meta.append({
                        'battery_id': battery_id,
                        'condition': condition,
                        'cycle': k + 1
                    })
                    
                print(f"  Completed {battery_id} in {time.time() - start_t:.2f} seconds. Valid cycles: {n_cycles}")
                
            except Exception as e:
                print(f"  Error processing {battery_id}: {str(e)}")
                
    # Save cycle-level features
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(os.path.join(output_dir, "cycle_features.csv"), index=False)
    print(f"\nSaved cycle-level features table of shape {df_features.shape} to cycle_features.csv")
    
    # Save LSTM sequences and targets
    lstm_sequences = np.array(lstm_sequences)  # Shape (N_total_cycles, 100, 3)
    np.save(os.path.join(output_dir, "lstm_sequences.npy"), lstm_sequences)
    
    df_lstm_targets = pd.DataFrame({
        'soh': lstm_targets_soh,
        'rul': lstm_targets_rul
    })
    df_lstm_targets.to_csv(os.path.join(output_dir, "lstm_targets.csv"), index=False)
    
    df_lstm_meta = pd.DataFrame(lstm_meta)
    df_lstm_meta.to_csv(os.path.join(output_dir, "lstm_meta.csv"), index=False)
    
    print(f"Saved LSTM sequences of shape {lstm_sequences.shape} to lstm_sequences.npy")
    print(f"Saved LSTM targets and metadata to CSV files.")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    process_all_data()
