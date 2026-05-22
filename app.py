import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import plotly.express as px
from src.anomaly_detection import detect_anomaly, calculate_health_score, get_flowchart_dot, BatteryEKF
import scipy.io

# Set page configuration
st.set_page_config(
    page_title="Battery Health Analytics System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1abc9c 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1e272e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2f3542;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1abc9c;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a4b0be;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Animation effects */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .pulse-card {
        animation: pulse 3s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find Knee Point (Kneedle algorithm)
def find_knee_point(cycles, soh):
    if len(cycles) < 5:
        return cycles[0], soh[0]
    p1 = np.array([cycles[0], soh[0]])
    p2 = np.array([cycles[-1], soh[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    distances = []
    for c, s in zip(cycles, soh):
        p = np.array([c, s])
        v = p - p1
        dist = np.linalg.norm(v - np.dot(v, line_unit) * line_unit)
        distances.append(dist)
        
    knee_idx = np.argmax(distances)
    return cycles[knee_idx], soh[knee_idx]

# Load Processed Data
processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
df_features = pd.read_csv(os.path.join(processed_dir, "cycle_features.csv"))
df_metrics = pd.read_csv(os.path.join(processed_dir, "model_comparison_metrics.csv"))
df_predictions = pd.read_csv(os.path.join(processed_dir, "test_predictions.csv"))
df_sim = pd.read_csv(os.path.join(processed_dir, "charging_simulation.csv"))

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/nolan/96/battery.png", width=80)
st.sidebar.markdown("<h2 style='font-family:Space Grotesk; font-weight:700;'>Battery Health</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#a4b0be; font-size:0.85rem; margin-top:-10px;'>NASA Randomized Dataset Framework</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 System Overview & Preprocessing",
        "⚙️ Feature Engineering & Physics",
        "🤖 Machine Learning Models",
        "🚨 Health & Anomaly Center",
        "⚡ Intelligent Charging Simulation",
        "🔮 Upload & Predict: Custom Diagnostics"
    ]
)

# ----------------- PAGE 1: OVERVIEW & PREPROCESSING -----------------
if page == "📊 System Overview & Preprocessing":
    st.markdown("<h1 class='main-title'>Battery Health Analytics System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>An Integrated Degradation-Aware Framework for Prediction, Anomaly Detection, and Management</p>", unsafe_allow_html=True)
    
    # Metadata metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='card pulse-card'>
            <div class='metric-label'>Total Batteries Analyzed</div>
            <div class='metric-value'>12 Cells</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='metric-label'>Total Cleaned Cycles</div>
            <div class='metric-value'>{len(df_features)} Cycles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card'>
            <div class='metric-label'>Dataset Source</div>
            <div class='metric-value' style='font-size:1.6rem; padding-top:10px;'>NASA Randomized</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='card'>
            <div class='metric-label'>Framework Tech Stack</div>
            <div class='metric-value' style='font-size:1.6rem; padding-top:10px;'>TF, Scikit, Streamlit</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📋 NASA Randomized Battery Usage Conditions")
    st.write("This research integrates all **7 randomized operational profiles** under different conditions:")
    
    cond_df = pd.DataFrame({
        'Condition Name': [
            'Uniform Room Temperature (RT)', 
            'Skewed Room Temperature (RT)', 
            'Skewed High Temperature (40°C)'
        ],
        'Batteries': ['RW9, RW10, RW11, RW12', 'RW17, RW18, RW19, RW20', 'RW25, RW26, RW27, RW28'],
        'Description': [
            'Cycled at 25°C ambient temperature with random walk current profiles centered on 0A.',
            'Cycled at 25°C ambient temperature with random walk current profiles skewed towards discharging.',
            'Cycled at 40°C ambient temperature with random walk current profiles skewed towards discharging.'
        ],
        'Reference Cycles': [
            'Reference charge/discharge after every 1500 RW steps to evaluate Capacity and SOH.',
            'Reference charge/discharge after every 1500 RW steps to evaluate Capacity and SOH.',
            'Reference charge/discharge after every 1500 RW steps to evaluate Capacity and SOH.'
        ]
    })
    st.table(cond_df)
    
    st.markdown("### 🔍 Raw vs Cleaned Reference Discharge Curves")
    st.write("Select a battery cell to visualize the raw vs cleaned capacity degradation profile across cycles:")
    
    selected_battery = st.selectbox("Select Battery ID", sorted(df_features['battery_id'].unique()))
    df_bat = df_features[df_features['battery_id'] == selected_battery]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_bat['cycle'], y=df_bat['capacity'],
        mode='lines+markers', name='Cleaned Capacity (Ah)',
        line=dict(color='#1abc9c', width=2),
        marker=dict(size=5)
    ))
    fig.update_layout(
        title=f"Capacity Fade Trend for Cell {selected_battery} ({df_bat['condition'].iloc[0]})",
        xaxis_title="Reference Discharge Cycle",
        yaxis_title="Discharge Capacity (Ah)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- PAGE 2: FEATURE ENGINEERING & PHYSICS -----------------
elif page == "⚙️ Feature Engineering & Physics":
    st.markdown("<h1 class='main-title'>Feature Engineering & Physics-based Degradation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Extraction of State of Health (SOH) and Internal Resistance (Rᵢ) per cycle using physical parameters</p>", unsafe_allow_html=True)
    
    selected_battery = st.selectbox("Select Battery ID to Analyze Physics Metrics", sorted(df_features['battery_id'].unique()))
    df_bat = df_features[df_features['battery_id'] == selected_battery].copy()
    
    # Kneedle Algorithm for Knee Point Detection
    knee_cycle, knee_soh = find_knee_point(df_bat['cycle'].values, df_bat['soh'].values)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 State of Health (SOH) Degradation Curve")
        st.write("The chart below shows SOH (%) with the detected **Knee Point (Kneedle Algorithm)** highlighted:")
        
        fig = go.Figure()
        # SOH line
        fig.add_trace(go.Scatter(
            x=df_bat['cycle'], y=df_bat['soh'],
            mode='lines', name='SOH (%)',
            line=dict(color='#3498db', width=3)
        ))
        # Knee point marker
        fig.add_trace(go.Scatter(
            x=[knee_cycle], y=[knee_soh],
            mode='markers+text', name='Knee Point (Degradation Acceleration)',
            marker=dict(color='#e74c3c', size=12, symbol='star'),
            text=[f"Knee Point: Cycle {knee_cycle}"],
            textposition="top right",
            textfont=dict(color='#ffffff')
        ))
        # EOL threshold line (80%)
        fig.add_shape(
            type="line", x0=df_bat['cycle'].min(), y0=80, x1=df_bat['cycle'].max(), y1=80,
            line=dict(color="#f1c40f", dash="dash", width=1.5)
        )
        fig.update_layout(
            xaxis_title="Cycle Index",
            yaxis_title="State of Health (SOH %)",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=20, b=20),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### 📈 Internal Resistance ($R_i$) Growth")
        st.write("Internal resistance is calculated per cycle via transient voltage drops using **Ohm's Law**: $R_i = \Delta V / \Delta I$")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_bat['cycle'], y=df_bat['ri'],
            mode='lines+markers', name='Internal Resistance (Ω)',
            line=dict(color='#e67e22', width=2),
            marker=dict(size=4)
        ))
        fig2.update_layout(
            xaxis_title="Cycle Index",
            yaxis_title="Internal Resistance (Ohms)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("### 📊 Extracted Cycle-Level Feature Table (First 10 Cycles)")
    st.dataframe(
        df_bat[['cycle', 'capacity', 'soh', 'rul', 'ri', 'peak_temp', 'voltage_drop', 'duration', 'mean_rw_temp']]
        .head(10)
        .style.format({
            'capacity': '{:.4f} Ah',
            'soh': '{:.2f}%',
            'rul': '{:d} cycles',
            'ri': '{:.4f} Ω',
            'peak_temp': '{:.2f} °C',
            'voltage_drop': '{:.3f} V',
            'duration': '{:.1f} s',
            'mean_rw_temp': '{:.2f} °C'
        })
    )

# ----------------- PAGE 3: MACHINE LEARNING MODELS -----------------
elif page == "🤖 Machine Learning Models":
    st.markdown("<h1 class='main-title'>Model Training & Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Performance metrics comparison of tabular Random Forest vs. sequence-based LSTM models</p>", unsafe_allow_html=True)
    
    # Display comparison metrics
    st.markdown("### 📊 Test Set Metrics Comparison (Strict Cross-Battery Split)")
    st.write("Evaluation metrics are calculated on unseen batteries (`RW12` Uniform RT, `RW20` Skewed RT, `RW28` Skewed High 40C):")
    st.dataframe(
        df_metrics.style.format({
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R2': '{:.4f}'
        })
    )
    
    # Model predictions vs actual plots
    st.markdown("### 📈 SOH and RUL Test Set Predictions")
    test_cell = st.selectbox("Select Test Cell for Visual Prediction Verification", df_predictions['battery_id'].unique())
    df_test_cell = df_predictions[df_predictions['battery_id'] == test_cell]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### SOH Prediction Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_actual_soh'], name='Actual SOH', line=dict(color='#ffffff', width=2)))
        fig.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_pred_rf_soh'], name='RF Predict (Tabular)', line=dict(color='#1abc9c', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_pred_lstm_soh'], name='LSTM Predict (Raw Sequence)', line=dict(color='#3498db', width=2, dash='dot')))
        fig.update_layout(xaxis_title="Cycle Index", yaxis_title="SOH (%)", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### RUL Prediction Comparison")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_actual_rul'], name='Actual RUL', line=dict(color='#ffffff', width=2)))
        fig2.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_pred_rf_rul'], name='RF Predict (Tabular)', line=dict(color='#1abc9c', width=2, dash='dash')))
        fig2.add_trace(go.Scatter(x=df_test_cell['cycle'], y=df_test_cell['y_pred_lstm_rul'], name='LSTM Predict (Raw Sequence)', line=dict(color='#3498db', width=2, dash='dot')))
        fig2.update_layout(xaxis_title="Cycle Index", yaxis_title="RUL (Cycles)", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig2, use_container_width=True)

# ----------------- PAGE 4: HEALTH & ANOMALY CENTER -----------------
elif page == "🚨 Health & Anomaly Center":
    st.markdown("<h1 class='main-title'>Battery Health Monitoring & Anomaly Center</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Real-time cell state monitoring, alarm triggers, and mitigation action flowchart</p>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎛️ Static Cell Simulator", "⚡ Real-Time BMS Monitor (EKF)"])
    
    with tab1:
        st.write("Adjust the sliders below to simulate different battery conditions and see the calculated Health Score and Alarms:")
        col1, col2 = st.columns(2)
        with col1:
            sim_soh = st.slider("State of Health (SOH %)", 30.0, 100.0, 88.0, step=0.5)
            sim_temp = st.slider("Cell Temperature (°C)", 20.0, 55.0, 28.0, step=0.5)
            sim_rul = st.slider("Predicted Remaining Useful Life (RUL)", 0, 150, 120, step=1)
            
        with col2:
            # Calculate health score and anomaly status
            health = calculate_health_score(sim_soh, sim_rul)
            anomaly = detect_anomaly(sim_soh, sim_temp)
            
            # Display large status cards
            st.markdown(f"""
            <div style='background-color:{health['color']}; border-radius:12px; padding:1.2rem; color:#1e272e; margin-bottom:1rem; font-weight:700;'>
                <div style='font-size:0.9rem; text-transform:uppercase; letter-spacing:0.5px;'>Composite Health Score</div>
                <div style='font-size:3.2rem; font-family:Space Grotesk;'>{health['score']}/100</div>
                <div style='font-size:1.2rem; margin-top:0.3rem;'>Condition Band: {health['band']}</div>
                <div style='font-size:0.85rem; font-weight:400; margin-top:0.2rem; opacity:0.9;'>{health['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color:#1e272e; border: 2px solid {anomaly['color']}; border-radius:12px; padding:1.2rem; margin-bottom:1rem;'>
                <div style='font-size:0.9rem; color:#a4b0be; text-transform:uppercase; letter-spacing:0.5px;'>Anomaly Status</div>
                <div style='font-size:2.2rem; font-weight:700; color:{anomaly['color']};'>{anomaly['status']} (Level {anomaly['level']})</div>
                <div style='font-size:0.9rem; color:#ffffff; margin-top:0.4rem;'><b>Potential Risk:</b> {anomaly['risk']}</div>
                <div style='font-size:0.9rem; color:#ffffff; margin-top:0.2rem;'><b>Mitigation Action:</b> {anomaly['mitigation']}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.write("Stream real-time battery sensor data to test the **Extended Kalman Filter (EKF)** and **Anomaly Detector**:")
        
        @st.cache_data
        def get_battery_raw_data(battery_id):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for folder in ["Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post", "RW_Skewed_High_Room_Temp_DataSet_2Post", "RW_Skewed_High_40C_DataSet_2Post"]:
                p = os.path.join(base_dir, folder, f"data/Matlab/{battery_id}.mat")
                if os.path.exists(p):
                    return scipy.io.loadmat(p)
            return None

        col_ctrl, col_stats = st.columns([1, 2])
        
        with col_ctrl:
            sel_bat = st.selectbox("Select Battery ID for Live Run", sorted(df_features['battery_id'].unique()), key="sel_bat_ekf")
            mat_data = get_battery_raw_data(sel_bat)
            
            is_cloud_mode = False
            if mat_data is None:
                is_cloud_mode = True
                st.warning("☁️ Cloud Mode: Raw MATLAB files not found. Using preloaded sample telemetry for Cell RW10 (Cycle 1).")
                valid_step_indices = [0]
                cycle_num = 1
                downsample = st.select_slider("Telemetry Downsampling Rate", options=[1, 2, 5, 10, 20], value=5)
                sim_noise = st.checkbox("Inject +50mA Sensor Bias (Test EKF Drift Rejection)", value=True)
                run_stream = st.checkbox("Start Live Telemetry Stream", value=False)
            else:
                step_struct = mat_data['data'][0, 0]['step']
                valid_step_indices = []
                for i in range(step_struct.shape[1]):
                    c = step_struct[0, i]['comment']
                    if c.size > 0 and str(c[0]) == 'reference discharge':
                        v_len = len(step_struct[0, i]['voltage'][0])
                        t_len = len(step_struct[0, i]['relativeTime'][0])
                        if v_len >= 50 and t_len >= 50:
                            valid_step_indices.append(i)
                
                cycle_num = st.selectbox("Select Discharge Cycle", range(1, len(valid_step_indices) + 1), index=0)
                downsample = st.select_slider("Telemetry Downsampling Rate", options=[1, 2, 5, 10, 20], value=5)
                sim_noise = st.checkbox("Inject +50mA Sensor Bias (Test EKF Drift Rejection)", value=True)
                run_stream = st.checkbox("Start Live Telemetry Stream", value=False)
                
        with col_stats:
            status_container = st.empty()
            chart_container = st.empty()
            
        if run_stream:
            if is_cloud_mode:
                csv_path = os.path.join(processed_dir, "sample_telemetry.csv")
                df_sample = pd.read_csv(csv_path)
                v_raw = df_sample['voltage'].values
                curr_raw = df_sample['current'].values
                temp_raw = df_sample['temperature'].values
                time_raw = df_sample['relativeTime'].values
                
                df_bat_feats = df_features[df_features['battery_id'] == "RW10"]
                current_soh = df_bat_feats['soh'].iloc[0] if len(df_bat_feats) > 0 else 88.0
                current_rul = int(df_bat_feats['rul'].iloc[0]) if len(df_bat_feats) > 0 else 120
                Q_n = df_bat_feats['capacity'].iloc[0] if len(df_bat_feats) > 0 else 2.0
                Q_actual = Q_n
            else:
                idx = valid_step_indices[cycle_num - 1]
                v_raw = step_struct[0, idx]['voltage'][0]
                curr_raw = step_struct[0, idx]['current'][0]
                temp_raw = np.clip(step_struct[0, idx]['temperature'][0], 15.0, 60.0)
                time_raw = step_struct[0, idx]['relativeTime'][0]
                
                df_bat_feats = df_features[df_features['battery_id'] == sel_bat]
                current_soh = df_bat_feats['soh'].iloc[cycle_num-1] if cycle_num-1 < len(df_bat_feats) else 80.0
                current_rul = int(df_bat_feats['rul'].iloc[cycle_num-1]) if cycle_num-1 < len(df_bat_feats) else 50
                Q_n = df_bat_feats['capacity'].iloc[0]
                Q_actual = df_bat_feats['capacity'].iloc[cycle_num-1] if cycle_num-1 < len(df_bat_feats) else Q_n
            
            # EKF starts at 90% SOC (10% mismatch to demonstrate initial convergence)
            ekf = BatteryEKF(R0=0.08, R1=0.05, C1=2000.0, Q_n=Q_actual, dt=1.0)
            ekf.x[0] = 0.90
            
            soc_cc_val = 0.90
            
            plot_time = []
            plot_true_soc = []
            plot_ekf_soc = []
            plot_cc_soc = []
            
            import time
            for i in range(0, len(time_raw), downsample):
                if i >= len(time_raw):
                    break
                    
                v_meas = v_raw[i]
                curr_true = curr_raw[i]
                curr_meas = curr_true + (0.05 if sim_noise else 0.0)
                temp_meas = temp_raw[i]
                t_val = time_raw[i]
                
                dt_step = time_raw[i] - time_raw[i - downsample] if i >= downsample else 1.0
                ekf.dt = dt_step
                ekf.predict(curr_meas)
                ekf.correct(v_meas, curr_meas)
                
                soc_cc_val = soc_cc_val - (dt_step / (3600.0 * Q_actual)) * curr_meas
                
                discharged_ah = np.sum(np.diff(time_raw[:i+1]) * (curr_raw[:i] + curr_raw[1:i+1]) / 2.0) / 3600.0 if i > 0 else 0.0
                soc_true_val = max(0.0, 1.0 - discharged_ah / Q_actual)
                
                plot_time.append(t_val)
                plot_true_soc.append(soc_true_val * 100.0)
                plot_ekf_soc.append(ekf.x[0] * 100.0)
                plot_cc_soc.append(soc_cc_val * 100.0)
                
                h_metric = calculate_health_score(current_soh, current_rul)
                a_metric = detect_anomaly(current_soh, temp_meas)
                
                status_container.markdown(f"""
                <div style='display:flex; gap:10px; margin-bottom:10px;'>
                    <div class='card' style='flex:1; margin-bottom:0; text-align:center; padding:0.8rem;'>
                        <div class='metric-label' style='font-size:0.75rem;'>Measured Voltage</div>
                        <div class='metric-value' style='font-size:1.5rem; color:#f1c40f;'>{v_meas:.3f} V</div>
                    </div>
                    <div class='card' style='flex:1; margin-bottom:0; text-align:center; padding:0.8rem;'>
                        <div class='metric-label' style='font-size:0.75rem;'>Current (True/Noisy)</div>
                        <div class='metric-value' style='font-size:1.4rem; color:#3498db;'>{curr_true:.2f}A / {curr_meas:.2f}A</div>
                    </div>
                    <div class='card' style='flex:1; margin-bottom:0; text-align:center; padding:0.8rem;'>
                        <div class='metric-label' style='font-size:0.75rem;'>Cell Temperature</div>
                        <div class='metric-value' style='font-size:1.5rem; color:{a_metric['color']};'>{temp_meas:.1f} °C</div>
                    </div>
                </div>
                <div style='display:flex; gap:10px; margin-bottom:10px;'>
                    <div style='flex:1; background-color:{h_metric['color']}; color:#1e272e; border-radius:12px; padding:0.8rem; text-align:center; font-weight:700;'>
                        <div style='font-size:0.75rem; text-transform:uppercase;'>Health Score</div>
                        <div style='font-size:1.8rem; font-family:Space Grotesk;'>{h_metric['score']}/100 ({h_metric['band']})</div>
                    </div>
                    <div style='flex:1.5; background-color:#1e272e; border: 2px solid {a_metric['color']}; border-radius:12px; padding:0.8rem;'>
                        <div style='font-size:0.75rem; color:#a4b0be; text-transform:uppercase;'>Anomaly: <span style='color:{a_metric['color']}; font-weight:700;'>{a_metric['status']}</span></div>
                        <div style='font-size:0.8rem; color:#ffffff; margin-top:2px;'><b>Mitigation:</b> {a_metric['mitigation']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(x=plot_time, y=plot_true_soc, name='True SOC (Ground Truth)', line=dict(color='#ffffff', width=2)))
                fig_live.add_trace(go.Scatter(x=plot_time, y=plot_ekf_soc, name='EKF Estimated SOC', line=dict(color='#2ecc71', width=2, dash='dash')))
                fig_live.add_trace(go.Scatter(x=plot_time, y=plot_cc_soc, name='Coulomb Counting (Drifting)', line=dict(color='#e74c3c', width=2, dash='dot')))
                fig_live.update_layout(
                    title="Real-Time State of Charge (SOC) Estimator Tracking",
                    xaxis_title="Time (s)",
                    yaxis_title="SOC (%)",
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300
                )
                chart_container.plotly_chart(fig_live, use_container_width=True)
                time.sleep(0.04)
                
            st.success("Telemetry Stream Completed!")
        
    st.markdown("---")
    col3, col4 = st.columns([2, 3])
    
    with col3:
        st.markdown("### 🗺️ Alarm Decision Flowchart")
        st.write("Decision tree logic for alarm classification and cutoffs:")
        st.graphviz_chart(get_flowchart_dot())
        
    with col4:
        st.markdown("### 📋 Active Mitigation Action Table")
        st.write("Fixed reference policy table for battery state monitoring and emergency handling:")
        
        action_df = pd.DataFrame({
            'System State': ['Normal (Level 1)', 'Warning (Level 2)', 'Critical (Level 3)'],
            'Trigger Conditions': [
                'SOH > 85% AND Temperature < 40°C',
                'SOH 70% to 85% OR Temperature >= 40°C',
                'SOH < 70%'
            ],
            'System Risk': [
                'Safe operational bounds, standard cell aging.',
                'Capacity loss/elevated heat dissipation, accelerated aging.',
                'High thermal runaway risk, internal short-circuit risk.'
            ],
            'Mitigation Actions': [
                'Log diagnostics, standard charge/discharge profiles.',
                'Cap charging current to 0.5C, trigger system warning alert.',
                'Immediate emergency cutoff, send push alarm to operator.'
            ]
        })
        st.table(action_df)

# ----------------- PAGE 5: INTELLIGENT CHARGING SIMULATION -----------------
elif page == "⚡ Intelligent Charging Simulation":
    st.markdown("<h1 class='main-title'>Intelligent Charging Simulation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Comparison of standard 1.0C charging vs. degradation-aware current capping (Scenario A vs B)</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div class='card pulse-card'>
            <div class='metric-label'>Simulated Cycle Lifetime Extension</div>
            <div class='metric-value' style='color:#2ecc71;'>+18.7%</div>
            <div style='font-size:0.85rem; color:#a4b0be; margin-top:5px;'>Scenario B extends cell EOL from cycle 107 to 127.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 💡 Scenario B Rule
        When internal resistance ($R_i$) rises **15% above the baseline** (i.e. $\ge 0.092\ \Omega$), the charging current is capped to **0.5C** (1.0A).
        
        This reduces cell internal heating ($I^2R_i$), lowers peak cell temperature, and significantly slows down SOH degradation!
        """)
        
    with col2:
        # SOH comparison plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['soh_A'], name='Scenario A (Standard)', line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['soh_B'], name='Scenario B (Degradation-Aware)', line=dict(color='#2ecc71', width=2)))
        # EOL line
        fig.add_shape(
            type="line", x0=df_sim['cycle'].min(), y0=80, x1=df_sim['cycle'].max(), y1=80,
            line=dict(color="#ffffff", dash="dash", width=1)
        )
        fig.update_layout(
            title="State of Health (SOH) Fade Trajectory Comparison",
            xaxis_title="Charge Cycle",
            yaxis_title="SOH (%)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        # Cell temperature plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['temp_A'], name='Scenario A Temp', line=dict(color='#e74c3c', width=1.5)))
        fig2.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['temp_B'], name='Scenario B Temp', line=dict(color='#2ecc71', width=1.5)))
        fig2.update_layout(
            title="Peak Temperature Profile per Charging Event",
            xaxis_title="Charge Cycle",
            yaxis_title="Cell Temperature (°C)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    with col4:
        # Charging current plot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['curr_A'], name='Scenario A Current', line=dict(color='#e74c3c', width=2)))
        fig3.add_trace(go.Scatter(x=df_sim['cycle'], y=df_sim['curr_B'], name='Scenario B Current', line=dict(color='#2ecc71', width=2)))
        fig3.update_layout(
            title="Applied Charge Current Profile",
            xaxis_title="Charge Cycle",
            yaxis_title="Charging Current (A)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=350
        )
        st.plotly_chart(fig3, use_container_width=True)

# ----------------- PAGE 6: CUSTOM DIAGNOSTICS UPLOAD & PREDICT -----------------
elif page == "🔮 Upload & Predict: Custom Diagnostics":
    st.markdown("<h1 class='main-title'>Custom Diagnostic Center</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload your own CSV or MAT files to run degradation diagnostics, health scoring, and SOH/RUL predictions.</p>", unsafe_allow_html=True)
    
    # Check if TensorFlow is available for sequence prediction
    try:
        import tensorflow as tf
        HAS_TENSORFLOW = True
    except ImportError:
        HAS_TENSORFLOW = False
        
    # Load ML Models from processed_dir
    @st.cache_resource
    def load_rf_models():
        with open(os.path.join(processed_dir, "rf_soh.pkl"), "rb") as f:
            rf_soh = pickle.load(f)
        with open(os.path.join(processed_dir, "rf_rul.pkl"), "rb") as f:
            rf_rul = pickle.load(f)
        return rf_soh, rf_rul

    @st.cache_resource
    def load_lstm_models():
        if not HAS_TENSORFLOW:
            return None, None, None
        soh_lstm = tf.keras.models.load_model(os.path.join(processed_dir, "lstm_soh.keras"))
        rul_lstm = tf.keras.models.load_model(os.path.join(processed_dir, "lstm_rul.keras"))
        with open(os.path.join(processed_dir, "seq_norm_params.pkl"), "rb") as f:
            norm_params = pickle.load(f)
        return soh_lstm, rul_lstm, norm_params

    try:
        rf_soh_model, rf_rul_model = load_rf_models()
        rf_loaded = True
    except Exception as e:
        st.error(f"Error loading Random Forest models: {e}")
        rf_loaded = False

    lstm_soh_model, lstm_rul_model, lstm_norm_params = load_lstm_models()

    if not HAS_TENSORFLOW:
        st.info("ℹ️ Running in lightweight mode (TensorFlow not detected). Sequence-based LSTM predictions are disabled, but tabular Random Forest models are fully active. Run the dashboard locally inside the virtual environment to enable full LSTM diagnostics.")

    tab1, tab2 = st.tabs(["📇 Upload Tabular Cycle Features (CSV)", "📊 Upload Raw Cycle Telemetry (CSV/MAT)"])

    with tab1:
        st.write("Upload a CSV file containing precalculated cycle-level parameters for your battery.")
        st.markdown("""
        **Required Columns:**
        - `cycle`: Cycle number (integer)
        - `ri`: Internal Resistance (Ohms, e.g., 0.08)
        - `peak_temp`: Peak temperature during discharge (°C)
        - `voltage_drop`: Total discharge voltage drop (V)
        - `duration`: Discharge event duration (seconds)
        - `mean_rw_temp`: Mean temperature during random walk steps (°C)
        - `max_rw_temp`: Maximum temperature during random walk steps (°C)
        """)

        # Provide a sample cycle feature template for download
        sample_tabular_df = pd.DataFrame([
            {'cycle': 1, 'ri': 0.081, 'peak_temp': 28.5, 'voltage_drop': 1.15, 'duration': 3500.0, 'mean_rw_temp': 24.8, 'max_rw_temp': 25.2},
            {'cycle': 10, 'ri': 0.083, 'peak_temp': 29.1, 'voltage_drop': 1.18, 'duration': 3420.0, 'mean_rw_temp': 25.1, 'max_rw_temp': 25.5},
            {'cycle': 50, 'ri': 0.091, 'peak_temp': 31.4, 'voltage_drop': 1.25, 'duration': 3100.0, 'mean_rw_temp': 24.9, 'max_rw_temp': 25.3},
            {'cycle': 100, 'ri': 0.112, 'peak_temp': 35.8, 'voltage_drop': 1.38, 'duration': 2650.0, 'mean_rw_temp': 25.0, 'max_rw_temp': 25.4}
        ])
        
        st.download_button(
            label="Download Tabular CSV Template",
            data=sample_tabular_df.to_csv(index=False),
            file_name="battery_features_template.csv",
            mime="text/csv"
        )

        uploaded_tabular = st.file_uploader("Choose Tabular CSV File", type=["csv"], key="uploader_tab1")

        if uploaded_tabular is not None and rf_loaded:
            try:
                df_up = pd.read_csv(uploaded_tabular)
                required_cols = ['cycle', 'ri', 'peak_temp', 'voltage_drop', 'duration', 'mean_rw_temp', 'max_rw_temp']
                
                missing_cols = []
                for c in required_cols:
                    match = [col for col in df_up.columns if col.strip().lower() == c]
                    if match:
                        df_up[c] = df_up[match[0]]
                    else:
                        missing_cols.append(c)
                
                if len(missing_cols) > 0:
                    st.error(f"Missing required columns in CSV: {missing_cols}")
                else:
                    df_up = df_up.sort_values('cycle').reset_index(drop=True)
                    
                    X_input = df_up[required_cols].values.astype(np.float32)
                    df_up['SOH_Prediction'] = rf_soh_model.predict(X_input)
                    df_up['RUL_Prediction'] = rf_rul_model.predict(X_input)
                    
                    st.success("Successfully run Random Forest Diagnostic Models!")
                    
                    latest_row = df_up.iloc[-1]
                    l_soh = latest_row['SOH_Prediction']
                    l_rul = latest_row['RUL_Prediction']
                    l_cycle = int(latest_row['cycle'])
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Latest Diagnosed Cycle", f"Cycle {l_cycle}")
                    with col_m2:
                        st.metric("Predicted SOH (%)", f"{l_soh:.2f}%")
                    with col_m3:
                        st.metric("Predicted RUL (Remaining Cycles)", f"{int(l_rul)} Cycles")
                        
                    fig_soh = go.Figure()
                    fig_soh.add_trace(go.Scatter(x=df_up['cycle'], y=df_up['SOH_Prediction'], name='SOH Forecast (%)', line=dict(color='#1abc9c', width=3)))
                    fig_soh.add_shape(type="line", x0=df_up['cycle'].min(), y0=80, x1=df_up['cycle'].max(), y1=80, line=dict(color="#f1c40f", dash="dash"))
                    fig_soh.update_layout(title="Diagnosed State of Health (SOH) Trend", xaxis_title="Cycle Index", yaxis_title="SOH (%)", template="plotly_dark", height=350)
                    st.plotly_chart(fig_soh, use_container_width=True)
                    
                    st.dataframe(df_up[['cycle', 'ri', 'peak_temp', 'voltage_drop', 'duration', 'SOH_Prediction', 'RUL_Prediction']].style.format({
                        'ri': '{:.4f} Ω',
                        'peak_temp': '{:.2f} °C',
                        'voltage_drop': '{:.3f} V',
                        'duration': '{:.1f} s',
                        'SOH_Prediction': '{:.2f}%',
                        'RUL_Prediction': '{:.0f} cycles'
                    }))
            except Exception as e:
                st.error(f"Error executing tabular diagnostics: {e}")

    with tab2:
        st.write("Upload a raw charge-discharge cycle time-series file. This can be a CSV containing time-series columns or a MATLAB (.mat) step-structure file.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("""
            **Expected CSV Columns (Raw Time-Series):**
            - `relativeTime` (or `time`): Seconds from start (e.g. 0, 1, 2...)
            - `voltage` (or `V`): Cell voltage (V, e.g. 4.2 to 2.7)
            - `current` (or `I`): Cell current (A, discharge is negative, charge is positive)
            - `temperature` (or `T`): Cell temperature (°C)
            """)
        with col_c2:
            st.markdown("""
            **Expected MAT Format:**
            - A MATLAB workspace export containing the NASA `data` workspace variable with a nested `step` structure.
            """)

        # Provide template download for raw CSV
        sample_raw_df = pd.read_csv(os.path.join(processed_dir, "sample_telemetry.csv"))
        st.download_button(
            label="Download Raw Telemetry CSV Template",
            data=sample_raw_df.to_csv(index=False),
            file_name="battery_raw_telemetry_template.csv",
            mime="text/csv"
        )

        uploaded_raw = st.file_uploader("Choose Raw Telemetry File (CSV or MAT)", type=["csv", "mat"], key="uploader_tab2")
        custom_cycle_idx = st.number_input("Specify Current Battery Cycle Index", min_value=1, value=1, step=1)

        if uploaded_raw is not None:
            v_raw, curr_raw, temp_raw, time_raw = None, None, None, None
            
            filename = uploaded_raw.name
            if filename.endswith('.mat'):
                try:
                    import io
                    mat_data = scipy.io.loadmat(io.BytesIO(uploaded_raw.read()))
                    if 'data' in mat_data:
                        step_struct = mat_data['data'][0, 0]['step']
                        found_dis = False
                        for idx in range(step_struct.shape[1]):
                            c = step_struct[0, idx]['comment']
                            if c.size > 0 and str(c[0]) == 'reference discharge':
                                v_raw = step_struct[0, idx]['voltage'][0]
                                curr_raw = step_struct[0, idx]['current'][0]
                                temp_raw = np.clip(step_struct[0, idx]['temperature'][0], 15.0, 60.0)
                                time_raw = step_struct[0, idx]['relativeTime'][0]
                                if len(v_raw) >= 50 and len(time_raw) >= 50:
                                    found_dis = True
                                    st.info(f"Loaded reference discharge step index {idx} from MAT file.")
                                    break
                        if not found_dis:
                            st.error("Could not find any valid reference discharge steps inside the MAT step structure.")
                    else:
                        st.error("MAT workspace doesn't contain a 'data' struct.")
                except Exception as e:
                    st.error(f"Error parsing uploaded MAT file: {e}")
            else:
                try:
                    df_raw = pd.read_csv(uploaded_raw)
                    cols = [c.lower().strip() for c in df_raw.columns]
                    
                    v_match = [col for col in df_raw.columns if col.strip().lower() in ['voltage', 'v']]
                    curr_match = [col for col in df_raw.columns if col.strip().lower() in ['current', 'i', 'curr']]
                    temp_match = [col for col in df_raw.columns if col.strip().lower() in ['temperature', 't', 'temp']]
                    time_match = [col for col in df_raw.columns if col.strip().lower() in ['relativetime', 'time', 't_s']]
                    
                    if not v_match or not curr_match:
                        st.error("CSV must contain at least 'voltage' and 'current' columns.")
                    else:
                        v_raw = df_raw[v_match[0]].values
                        curr_raw = df_raw[curr_match[0]].values
                        temp_raw = df_raw[temp_match[0]].values if temp_match else np.ones_like(v_raw) * 25.0
                        time_raw = df_raw[time_match[0]].values if time_match else np.arange(len(v_raw))
                        st.info("Successfully loaded CSV columns.")
                except Exception as e:
                    st.error(f"Error reading CSV raw telemetry: {e}")

            if v_raw is not None and curr_raw is not None and time_raw is not None:
                capacity = np.sum(np.diff(time_raw) * (curr_raw[:-1] + curr_raw[1:]) / 2.0) / 3600.0 if len(time_raw) > 1 else 0.0
                ri = 0.08
                if len(curr_raw) > 1:
                    dv = v_raw[0] - v_raw[1]
                    di = curr_raw[1] - curr_raw[0]
                    if abs(di) > 0.01:
                        ri = dv / di
                if ri <= 0.01 or ri > 0.5:
                    ri = 0.08
                
                peak_temp = np.max(temp_raw)
                voltage_drop = v_raw[0] - v_raw[-1]
                duration = time_raw[-1] - time_raw[0]
                
                st.success("Physical Features Extracted Successfully!")
                
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                with col_e1:
                    st.metric("Discharge Capacity", f"{capacity:.4f} Ah")
                with col_e2:
                    st.metric("Internal Resistance", f"{ri:.4f} Ω")
                with col_e3:
                    st.metric("Peak Temperature", f"{peak_temp:.2f} °C")
                with col_e4:
                    st.metric("Voltage Drop", f"{voltage_drop:.3f} V")

                X_rf = np.array([[custom_cycle_idx, ri, peak_temp, voltage_drop, duration, 25.0, 25.0]], dtype=np.float32)
                pred_soh_rf = rf_soh_model.predict(X_rf)[0]
                pred_rul_rf = rf_rul_model.predict(X_rf)[0]

                lstm_ran = False
                if HAS_TENSORFLOW and lstm_soh_model is not None:
                    try:
                        xp = np.linspace(0, 1, len(v_raw))
                        x_new = np.linspace(0, 1, 100)
                        v_100 = np.interp(x_new, xp, v_raw)
                        curr_100 = np.interp(x_new, xp, curr_raw)
                        temp_100 = np.interp(x_new, xp, temp_raw)
                        
                        seq = np.column_stack([v_100, curr_100, temp_100])
                        seq_scaled = (seq - lstm_norm_params['mean']) / lstm_norm_params['std']
                        seq_scaled = np.expand_dims(seq_scaled, axis=0).astype(np.float32)
                        
                        pred_soh_lstm = lstm_soh_model.predict(seq_scaled, verbose=0)[0][0]
                        pred_rul_lstm = lstm_rul_model.predict(seq_scaled, verbose=0)[0][0]
                        lstm_ran = True
                    except Exception as e:
                        st.warning(f"Failed running sequence-based LSTM: {e}")

                st.markdown("### 🔮 Diagnostics Results")
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_soh = go.Figure()
                    fig_soh.add_trace(go.Bar(name='Random Forest', x=['SOH (%)'], y=[pred_soh_rf], marker_color='#1abc9c'))
                    if lstm_ran:
                        fig_soh.add_trace(go.Bar(name='LSTM (Sequence)', x=['SOH (%)'], y=[pred_soh_lstm], marker_color='#3498db'))
                    fig_soh.update_layout(yaxis_range=[30, 100], title="Predicted State of Health (SOH)", template="plotly_dark", height=300)
                    st.plotly_chart(fig_soh, use_container_width=True)
                    
                with col_d2:
                    fig_rul = go.Figure()
                    fig_rul.add_trace(go.Bar(name='Random Forest', x=['RUL (Cycles)'], y=[pred_rul_rf], marker_color='#e67e22'))
                    if lstm_ran:
                        fig_rul.add_trace(go.Bar(name='LSTM (Sequence)', x=['RUL (Cycles)'], y=[pred_rul_lstm], marker_color='#e74c3c'))
                    fig_rul.update_layout(title="Predicted Remaining Useful Life (RUL)", template="plotly_dark", height=300)
                    st.plotly_chart(fig_rul, use_container_width=True)

                anomaly_state = detect_anomaly(pred_soh_rf, peak_temp)
                health_state = calculate_health_score(pred_soh_rf, pred_rul_rf)
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"""
                    <div style='background-color:{health_state['color']}; border-radius:12px; padding:1.2rem; color:#1e272e; font-weight:700;'>
                        <div style='font-size:0.8rem; text-transform:uppercase;'>Composite Health Score</div>
                        <div style='font-size:2.8rem; font-family:Space Grotesk;'>{health_state['score']}/100</div>
                        <div style='font-size:1.1rem; margin-top:0.3rem;'>Condition Band: {health_state['band']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_s2:
                    st.markdown(f"""
                    <div style='background-color:#1e272e; border: 2px solid {anomaly_state['color']}; border-radius:12px; padding:1.2rem;'>
                        <div style='font-size:0.8rem; color:#a4b0be; text-transform:uppercase;'>Anomaly Status</div>
                        <div style='font-size:2.0rem; font-weight:700; color:{anomaly_state['color']};'>{anomaly_state['status']}</div>
                        <div style='font-size:0.85rem; color:#ffffff; margin-top:0.4rem;'><b>Action:</b> {anomaly_state['mitigation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
