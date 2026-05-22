import sys
import os
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.electrochemistry import calculate_ica, calculate_dva, fit_ecm_parameters

def main():
    print("Testing electrochemistry diagnostics module...")
    
    # Load sample telemetry
    telemetry_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_processed/sample_telemetry.csv'))
    if not os.path.exists(telemetry_path):
        print(f"Error: Sample telemetry file not found at {telemetry_path}")
        return
        
    df = pd.read_csv(telemetry_path)
    print(f"Loaded {len(df)} rows of sample telemetry.")
    
    # Check columns
    print("Columns:", df.columns.tolist())
    
    # Calculate artificial capacity Q from current and time for testing ICA/DVA
    # Q = integral of current dt
    dt = np.diff(df['relativeTime'].values)
    current = df['current'].values
    capacity = np.zeros(len(df))
    for i in range(1, len(df)):
        capacity[i] = capacity[i-1] + abs(current[i-1]) * dt[i-1] / 3600.0
        
    voltage = df['voltage'].values
    time = df['relativeTime'].values
    
    # 1. ICA
    V_mid, dq_dv = calculate_ica(voltage, capacity)
    print(f"ICA: V_mid size = {len(V_mid)}, dq_dv size = {len(dq_dv)}")
    if len(dq_dv) > 0:
        print(f"ICA range: min={dq_dv.min():.4f}, max={dq_dv.max():.4f}")
        
    # 2. DVA
    Q_mid, dv_dq = calculate_dva(voltage, capacity)
    print(f"DVA: Q_mid size = {len(Q_mid)}, dv_dq size = {len(dv_dq)}")
    if len(dv_dq) > 0:
        print(f"DVA range: min={dv_dq.min():.4f}, max={dv_dq.max():.4f}")
        
    # 3. ECM Parameter Fitting
    ecm = fit_ecm_parameters(voltage, current, time)
    print("ECM parameters:", ecm)
    
    assert ecm['R0'] > 0.0, "R0 must be positive"
    assert ecm['R1'] > 0.0, "R1 must be positive"
    assert ecm['C1'] > 0.0, "C1 must be positive"
    print("All tests passed successfully!")

if __name__ == "__main__":
    main()
