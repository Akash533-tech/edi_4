import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def calculate_ica(voltage, capacity, window_len=15, polyorder=2):
    """
    Computes smoothed Incremental Capacity (dQ/dV) from voltage and capacity.
    Filters voltage to be strictly monotonic to avoid divide-by-zero.
    """
    # Sort and filter duplicate voltages
    df = pd.DataFrame({'V': voltage, 'Q': capacity})
    df = df.sort_values(by='V').drop_duplicates(subset=['V']).reset_index(drop=True)
    
    if len(df) < 10:
        return np.array([]), np.array([])
        
    # Apply Savitzky-Golay filter to smooth Q and V
    try:
        w_len = min(window_len, len(df) - 1)
        if w_len % 2 == 0:
            w_len -= 1
        w_len = max(5, w_len)
        
        V_smooth = savgol_filter(df['V'].values, w_len, polyorder)
        Q_smooth = savgol_filter(df['Q'].values, w_len, polyorder)
    except Exception:
        V_smooth = df['V'].values
        Q_smooth = df['Q'].values

    # Calculate derivative
    dV = np.diff(V_smooth)
    dQ = np.diff(Q_smooth)
    
    # Avoid divide by zero
    dV = np.where(np.abs(dV) < 1e-5, np.sign(dV)*1e-5, dV)
    
    dq_dv = dQ / dV
    V_mid = (V_smooth[:-1] + V_smooth[1:]) / 2.0
    
    # Smooth dq_dv to make plots look high-quality
    try:
        w_len_dq = min(window_len, len(dq_dv) - 1)
        if w_len_dq % 2 == 0:
            w_len_dq -= 1
        w_len_dq = max(5, w_len_dq)
        dq_dv_smooth = savgol_filter(dq_dv, w_len_dq, polyorder)
    except Exception:
        dq_dv_smooth = dq_dv
        
    return V_mid, dq_dv_smooth

def calculate_dva(voltage, capacity, window_len=15, polyorder=2):
    """
    Computes smoothed Differential Voltage (dV/dQ) from voltage and capacity.
    """
    df = pd.DataFrame({'V': voltage, 'Q': capacity})
    df = df.sort_values(by='Q').drop_duplicates(subset=['Q']).reset_index(drop=True)
    
    if len(df) < 10:
        return np.array([]), np.array([])
        
    try:
        w_len = min(window_len, len(df) - 1)
        if w_len % 2 == 0:
            w_len -= 1
        w_len = max(5, w_len)
        
        V_smooth = savgol_filter(df['V'].values, w_len, polyorder)
        Q_smooth = savgol_filter(df['Q'].values, w_len, polyorder)
    except Exception:
        V_smooth = df['V'].values
        Q_smooth = df['Q'].values

    dV = np.diff(V_smooth)
    dQ = np.diff(Q_smooth)
    
    dQ = np.where(np.abs(dQ) < 1e-5, np.sign(dQ)*1e-5, dQ)
    
    dv_dq = dV / dQ
    Q_mid = (Q_smooth[:-1] + Q_smooth[1:]) / 2.0
    
    try:
        w_len_dv = min(window_len, len(dv_dq) - 1)
        if w_len_dv % 2 == 0:
            w_len_dv -= 1
        w_len_dv = max(5, w_len_dv)
        dv_dq_smooth = savgol_filter(dv_dq, w_len_dv, polyorder)
    except Exception:
        dv_dq_smooth = dv_dq
        
    return Q_mid, dv_dq_smooth

def fit_ecm_parameters(voltage, current, time):
    """
    Fits a 1-RC Thevenin circuit parameters (R0, R1, C1) from a raw cycle profile.
    If a relaxation (rest) phase is detected (current is 0 after discharge), it fits the recovery curve.
    Otherwise, it estimates parameters from load transients.
    """
    time = np.array(time)
    voltage = np.array(voltage)
    current = np.array(current)
    
    if len(time) < 10:
        return {'R0': 0.08, 'R1': 0.02, 'C1': 1500.0, 'tau': 30.0}
        
    R0 = 0.08
    R1 = 0.02
    C1 = 1500.0
    
    # Try relaxation phase fitting
    rest_indices = np.where(np.abs(current) < 0.01)[0]
    dis_indices = np.where(current < -0.1)[0]
    
    if len(rest_indices) > 20 and len(dis_indices) > 20:
        transition_idx = -1
        for idx in range(len(current) - 1):
            if current[idx] < -0.1 and abs(current[idx+1]) < 0.01:
                transition_idx = idx
                break
        
        if transition_idx != -1 and len(current) - transition_idx > 10:
            V_before = voltage[transition_idx]
            V_after = voltage[transition_idx + 1]
            I_dis = -current[transition_idx]
            
            if I_dis > 0.1:
                R0 = max(0.01, min(0.5, (V_after - V_before) / I_dis))
            
            t_rest = time[transition_idx+1:] - time[transition_idx+1]
            V_rest = voltage[transition_idx+1:]
            
            try:
                Voc = V_rest[-1]
                y = np.clip(Voc - V_rest, 1e-6, None)
                valid_fit_mask = (y > 1e-4) & (t_rest < 300)
                if np.sum(valid_fit_mask) > 5:
                    t_fit = t_rest[valid_fit_mask]
                    ln_y = np.log(y[valid_fit_mask])
                    slope, intercept = np.polyfit(t_fit, ln_y, 1)
                    
                    tau = -1.0 / slope if slope < -1e-4 else 50.0
                    tau = max(5.0, min(600.0, tau))
                    
                    R1_est = max(0.005, min(0.3, (Voc - V_after) / I_dis))
                    C1_est = max(100.0, min(10000.0, tau / R1_est))
                    
                    return {'R0': float(R0), 'R1': float(R1_est), 'C1': float(C1_est), 'tau': float(tau)}
            except Exception:
                pass
                
    # Fallback step change estimation
    try:
        for idx in range(len(current) - 1):
            if abs(current[idx]) < 0.05 and current[idx+1] < -0.5:
                dV = voltage[idx] - voltage[idx+1]
                dI = -current[idx+1]
                R0 = max(0.02, min(0.3, dV / dI))
                break
    except Exception:
        pass
        
    return {'R0': float(R0), 'R1': float(R1), 'C1': float(C1), 'tau': float(R1 * C1)}
