import numpy as np

def detect_anomaly(soh, temp):
    """
    3-level anomaly detection rules:
    - Level 1 Normal: SOH > 85% AND Temp < 40°C
    - Level 2 Warning: SOH 70-85% OR Temp >= 40°C -> reduce load, trigger alert
    - Level 3 Critical: SOH < 70% -> emergency cutoff, user notification
    """
    if soh < 70.0:
        level = 3
        status = "Critical"
        risk = "Thermal runway risk, internal degradation high, cell close to failure."
        mitigation = "Emergency cutoff. Disable charging/discharging immediately. Send push alert to operator."
        color = "#e74c3c"
    elif (soh <= 85.0) or (temp >= 40.0):
        level = 2
        status = "Warning"
        risk = "Increased heat dissipation / significant capacity loss. High wear rate."
        mitigation = "Reduce load demand. Cap charge/discharge current to 0.5C. Log high temp/wear event."
        color = "#f39c12"
    else:
        level = 1
        status = "Normal"
        risk = "Cell operating within safe physical and thermal bounds."
        mitigation = "Continue standard operations. Log health parameters at regular intervals."
        color = "#2ecc71"
        
    return {
        'level': level,
        'status': status,
        'risk': risk,
        'mitigation': mitigation,
        'color': color
    }

def calculate_health_score(soh, rul, max_rul=150):
    """
    Composite Health Score (0–100):
    Health Score = 0.6 * SOH + 0.4 * (RUL / RUL_max * 100) - Anomaly_Penalty
    Anomaly Penalty: Normal = 0, Warning = 15, Critical = 50.
    """
    anomaly_info = detect_anomaly(soh, 25.0)  # Baseline temperature for penalty calculation
    # Override anomaly detection with active temp if provided
    # But for static score we can use temperature-dependent penalty
    
    # Calculate penalty
    if anomaly_info['level'] == 3:
        penalty = 50.0
    elif anomaly_info['level'] == 2:
        penalty = 15.0
    else:
        penalty = 0.0
        
    # Scale RUL term
    scaled_rul = (rul / max_rul) * 100.0 if max_rul > 0 else 0.0
    
    score = (0.6 * soh) + (0.4 * scaled_rul) - penalty
    score = np.clip(score, 0.0, 100.0)
    
    if score >= 80.0:
        band = "Healthy"
        color = "#2ecc71"
        desc = "The battery pack is in excellent condition. Operating efficiency is high."
    elif score >= 50.0:
        band = "Monitor"
        color = "#f39c12"
        desc = "Moderate degradation observed. Keep a check on charge current and temperature."
    else:
        band = "Replace"
        color = "#e74c3c"
        desc = "Severe degradation or critical anomaly detected. Battery requires immediate replacement."
        
    return {
        'score': round(score, 2),
        'band': band,
        'color': color,
        'description': desc
    }

def get_flowchart_dot():
    """
    Returns a Graphviz DOT representation of the 3-level action plan flowchart.
    """
    dot_code = """
    digraph G {
        graph [rankdir=TB, bgcolor="transparent"]
        node [shape=box, style="filled,rounded", fontname="Outfit", fontsize=11, color="#2c3e50", fontcolor="#ffffff", fillcolor="#34495e", penwidth=2]
        edge [fontname="Outfit", fontsize=9, color="#7f8c8d", penwidth=1.5]
        
        Start [label="Monitor Cell State\\n(SOH, Temperature)", fillcolor="#3498db", color="#2980b9"]
        
        CheckSOH [label="SOH < 70%?", shape=diamond, fillcolor="#2c3e50"]
        CheckWarning [label="SOH <= 85% OR\\nTemp >= 40°C?", shape=diamond, fillcolor="#2c3e50"]
        
        Critical [label="Level 3: Critical\\nEmergency Cutoff!\\nOperator Alerted", fillcolor="#e74c3c", color="#c0392b"]
        Warning [label="Level 2: Warning\\nCap Current to 0.5C\\nTrigger System Alarm", fillcolor="#f39c12", color="#d35400"]
        Normal [label="Level 1: Normal\\nStandard Operation\\nLog Data", fillcolor="#2ecc71", color="#27ae60"]
        
        Start -> CheckSOH
        CheckSOH -> Critical [label=" Yes"]
        CheckSOH -> CheckWarning [label=" No"]
        CheckWarning -> Warning [label=" Yes"]
        CheckWarning -> Normal [label=" No"]
    }
    """
    return dot_code

def get_ocv(soc):
    """
    OCV-SOC curve using a 5th-order polynomial fitted for LiCoO2 cell.
    SOC range [0, 1]. Returns OCV in volts.
    """
    s = np.clip(soc, 0.0, 1.0)
    return 3.14 + 1.15 * s - 1.18 * (s**2) + 2.52 * (s**3) - 1.62 * (s**4)

def get_docv_dsoc(soc):
    """
    Analytic derivative dOCV/dSOC of the OCV-SOC polynomial.
    """
    s = np.clip(soc, 0.0, 1.0)
    return 1.15 - 2.36 * s + 7.56 * (s**2) - 6.48 * (s**3)

class BatteryEKF:
    def __init__(self, R0=0.08, R1=0.05, C1=2000.0, Q_n=2.0, dt=1.0):
        # Parameters
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.Q_n = Q_n  # Nominal capacity in Ah
        self.dt = dt
        
        # State vector: [SOC, V_RC]^T
        self.x = np.array([1.0, 0.0])  # Start fully charged
        
        # Error covariance matrix P
        self.P = np.diag([0.1, 0.1])
        
        # Process noise covariance Q
        self.Q = np.diag([1e-6, 1e-5])
        
        # Measurement noise covariance R
        self.R = 0.01 ** 2  # Standard deviation of 10mV
        
    def predict(self, current):
        """
        EKF Time Update (Predict step).
        current is positive for discharging, negative for charging.
        """
        # State transition
        soc_prev, vrc_prev = self.x[0], self.x[1]
        
        # Integrate current for SOC
        soc_new = soc_prev - (self.dt / (3600.0 * self.Q_n)) * current
        
        # Transient RC voltage update
        alpha = np.exp(-self.dt / (self.R1 * self.C1))
        vrc_new = alpha * vrc_prev + self.R1 * (1.0 - alpha) * current
        
        # Update state vector
        self.x = np.array([np.clip(soc_new, 0.0, 1.0), vrc_new])
        
        # State Jacobian A
        A = np.array([
            [1.0, 0.0],
            [0.0, alpha]
        ])
        
        # Predict error covariance
        self.P = A @ self.P @ A.T + self.Q
        return self.x
        
    def correct(self, terminal_voltage, current):
        """
        EKF Measurement Update (Correct step).
        terminal_voltage is the measured cell terminal voltage.
        """
        soc, vrc = self.x[0], self.x[1]
        
        # Expected terminal voltage
        ocv = get_ocv(soc)
        v_est = ocv - vrc - current * self.R0
        
        # Measurement residual (error)
        e = terminal_voltage - v_est
        
        # Measurement Jacobian H = [dOCV/dSOC, -1]
        docv = get_docv_dsoc(soc)
        H = np.array([[docv, -1.0]])
        
        # Innovation covariance S
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain K
        K = self.P @ H.T / S[0, 0]
        
        # Correct state estimate
        self.x = self.x + K.flatten() * e
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        
        # Correct error covariance
        self.P = (np.eye(2) - K @ H) @ self.P
        
        return self.x
