import streamlit as st
import pandas as pd
import numpy as np
import os
# Load local .env file if it exists
if os.path.exists(".env"):
    try:
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val.strip()
    except Exception:
        pass
import pickle
import plotly.graph_objects as go
import plotly.express as px
from src.anomaly_detection import detect_anomaly, calculate_health_score, get_flowchart_dot, BatteryEKF
from src.electrochemistry import calculate_ica, calculate_dva, fit_ecm_parameters
from src.pdf_generator import generate_pdf_report
import scipy.io
from streamlit_option_menu import option_menu

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
    
    /* Dynamic centered responsive container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1500px;
    }
    
    /* Dynamic shifting gradient main title */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(-45deg, #1abc9c, #2ecc71, #3498db, #9b59b6);
        background-size: 400% 400%;
        animation: gradient-shift 10s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    /* Glowing card hover effect and ambient light reflection */
    .card {
        background-color: #111827;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #1f2937;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-4px);
        border-color: #1abc9c;
        box-shadow: 0 10px 25px -10px rgba(26, 188, 156, 0.35);
    }
    
    /* Ambient backdrop glow pointer effect */
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 100% 0%, rgba(26, 188, 156, 0.08) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1abc9c;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
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
    
    /* Sidebar charging battery logo animation styles */
    .sidebar-logo-container {
        padding: 1rem 0.5rem;
        text-align: center;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 1.5rem;
    }
    
    .battery-glow-icon {
        width: 48px;
        height: 24px;
        border: 2.5px solid #1abc9c;
        border-radius: 5px;
        position: relative;
        margin: 0 auto 12px auto;
        box-shadow: 0 0 12px rgba(26, 188, 156, 0.4);
        animation: battery-charge 4s infinite alternate ease-in-out;
    }
    
    .battery-glow-icon::after {
        content: '';
        position: absolute;
        top: 4px;
        right: -7px;
        width: 4px;
        height: 11px;
        background: #1abc9c;
        border-radius: 0 2px 2px 0;
    }
    
    @keyframes battery-charge {
        0% {
            background: linear-gradient(to right, #1abc9c 0%, transparent 0%);
            box-shadow: 0 0 4px rgba(26, 188, 156, 0.2);
        }
        100% {
            background: linear-gradient(to right, #1abc9c 100%, transparent 100%);
            box-shadow: 0 0 18px rgba(26, 188, 156, 0.95);
        }
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

def get_future_recommendations(soh, rul, ri, peak_temp):
    recs = []
    
    # 1. SOH-based Charging C-rate Capping
    if soh >= 90:
        recs.append({
            "category": "⚡ Charging Protocol",
            "action": "Maintain fast charging up to 1.0C. Cell exhibits low internal resistance.",
            "impact": "Optimal charge speed with negligible degradation impact."
        })
    elif soh >= 80:
        recs.append({
            "category": "⚡ Charging Protocol",
            "action": "Cap fast charging to 0.7C. Avoid consecutive fast charging cycles.",
            "impact": "Reduces mechanical stress on electrode structures by up to 25%."
        })
    else:
        recs.append({
            "category": "⚡ Charging Protocol",
            "action": "Enforce smart current capping to 0.5C maximum. Transition to slow-charge below 20% SOC.",
            "impact": "Minimizes lithium plating risk and local heating, extending remaining life by 18.7%."
        })
        
    # 2. Thermal Management Limits
    if peak_temp > 38.0 or ri > 0.12:
        recs.append({
            "category": "🌡️ Thermal Safeguards",
            "action": "Lower thermal shutdown limit to 40°C (normally 45°C). Trigger active cooling at 35°C.",
            "impact": "Prevents accelerated SEI layer growth and localized hot spots."
        })
    else:
        recs.append({
            "category": "🌡️ Thermal Safeguards",
            "action": "Standard thermal protection limit at 45°C is sufficient. Active cooling trigger at 38°C.",
            "impact": "Maintains chemical activity within standard operating window."
        })
        
    # 3. Depth of Discharge (DOD) Management
    if soh < 85:
        recs.append({
            "category": "🔋 Cycle Window",
            "action": "Restrict usable SOC window to 20% - 80% (soft-lock BMS limits). Avoid deep discharge cycles.",
            "impact": "Reduces structural mechanical cracking in cathodes and preserves remaining cycle capacity."
        })
    else:
        recs.append({
            "category": "🔋 Cycle Window",
            "action": "Standard operating window (10% - 90% SOC) is safe for current cell degradation state.",
            "impact": "Maximizes current range output."
        })
        
    # 4. Maintenance & Calibration schedule
    if rul < 20:
        recs.append({
            "category": "🔧 Calibration Schedule",
            "action": "Schedule physical resistance validation and check for gas swelling within 10 cycles.",
            "impact": "Prevents sudden cell failure or short-circuit hazards."
        })
    else:
        recs.append({
            "category": "🔧 Calibration Schedule",
            "action": "Perform full EKF sensor calibration and capacity re-identification in 50 cycles.",
            "impact": "Ensures SOC estimator remains accurate and avoids state tracking drift."
        })
    return recs

class BatteryAICopilot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.initialized = False
        
    def init_model(self):
        if self.initialized:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # Load DeepSeek-V4-Pro via HuggingFace Transformers
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Pro")
            self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4-Pro")
            self.initialized = True
        except Exception as e:
            self.initialized = False
            # Fallback generator handles queries gracefully

    def generate_hf(self, prompt, metrics, hf_token, model_name, chat_history=None):
        import requests
        
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        soh = metrics.get('soh', 85.0) if metrics else 85.0
        rul = metrics.get('rul', 150) if metrics else 150
        ri = metrics.get('ri', 0.08) if metrics else 0.08
        temp = metrics.get('temp', 25.0) if metrics else 25.0
        
        system_message = (
            f"You are a battery management system (BMS) AI assistant. "
            f"You are analyzing the current battery cell telemetry:\n"
            f"- State of Health (SOH): {soh:.2f}%\n"
            f"- Remaining Useful Life (RUL): {int(rul)} cycles\n"
            f"- Internal Resistance (RI): {ri:.4f} \u03a9\n"
            f"- Peak Operating Temperature: {temp:.1f}\u00b0C\n\n"
            f"Use this physical context to diagnose problems, recommend operational limits (such as charge current capping/C-rate limits), or suggest mitigations. "
            f"Be concise, engineering-focused, and professional in your answers."
        )
        
        messages = [{"role": "system", "content": system_message}]
        if chat_history:
            # Exclude the last user message to avoid duplication when appending current prompt
            for msg in chat_history[:-1]:
                role = msg["role"]
                if role in ["user", "assistant"]:
                    messages.append({"role": role, "content": msg["content"]})
                    
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res_data = response.json()
                msg_obj = res_data["choices"][0]["message"]
                content_val = msg_obj.get("content") or ""
                reasoning_val = msg_obj.get("reasoning_content") or ""
                
                full_text = ""
                if reasoning_val:
                    full_text += f"<think>\n{reasoning_val}\n</think>\n\n"
                full_text += content_val
                
                return full_text.strip() if full_text.strip() else "Error: Model returned empty content."
            else:
                return f"Error from Hugging Face Inference API (Status Code {response.status_code}): {response.text}"
        except Exception as e:
            return f"Error querying Hugging Face Inference API: {str(e)}"

    def generate(self, prompt, context_metrics=None, hf_token=None, model_name=None, chat_history=None):
        if hf_token and hf_token.strip():
            return self.generate_hf(prompt, context_metrics, hf_token.strip(), model_name, chat_history)

        if self.initialized and self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                # Decode only the newly generated tokens
                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                return generated.strip()
            except Exception:
                pass
        return self.fallback_generate(prompt, context_metrics)

    def fallback_generate(self, prompt, metrics):
        prompt_lower = prompt.lower()
        soh = metrics.get('soh', 85.0) if metrics else 85.0
        rul = metrics.get('rul', 150) if metrics else 150
        ri = metrics.get('ri', 0.08) if metrics else 0.08
        temp = metrics.get('temp', 25.0) if metrics else 25.0
        
        health_status = "Excellent" if soh >= 90 else "Moderate Degradation" if soh >= 80 else "Critical Degradation"
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return f"Hello! I am your DeepSeek-powered BMS AI Copilot. I have analyzed your cell telemetry:\n- **State of Health (SOH):** {soh:.2f}%\n- **Remaining Useful Life (RUL):** {int(rul)} cycles\n- **Internal Resistance (RI):** {ri:.4f} Ω\n- **Peak Operating Temp:** {temp:.1f}°C\n\nHow can I assist you with diagnostic insights, C-rate limits, or mitigations today?"
            
        elif "soh" in prompt_lower or "health" in prompt_lower:
            msg = f"Your cell is currently operating at **{soh:.2f}% State of Health** ({health_status}). "
            if soh < 80:
                msg += "This is below the standard EOL (End-of-Life) threshold of 80%. We are observing significant lithium plating risk. I highly recommend capping fast charging to 0.5C and restricting the active depth of discharge (DOD) to a 20%-80% SOC window to avoid dendrite shorts."
            elif soh < 90:
                msg += "Degradation is moderate. This is typically driven by solid electrolyte interphase (SEI) layer growth. Main recommendation is to cap fast charging at 0.7C and monitor peak temperatures during discharge cycles."
            else:
                msg += "The cell structural integrity is high. Standard fast charging and operating profiles are safe to continue."
            return msg
            
        elif "rul" in prompt_lower or "life" in prompt_lower:
            return f"The model forecasts a Remaining Useful Life of **{int(rul)} cycles** before capacity falls below the critical EOL capacity limit. Enforcing the proposed thermal threshold mitigation (under 40°C) could extend this by approximately 15-20 cycles."
            
        elif "anomal" in prompt_lower or "problem" in prompt_lower or "issue" in prompt_lower:
            if temp > 38.0 or ri > 0.12 or soh < 80:
                return f"**Diagnostic Alert:** Critical operational risks detected:\n1. **Thermal Stress**: High peak discharge temperatures ({temp:.2f}°C) accelerate capacity fading.\n2. **Resistance Spikes**: Internal resistance is at {ri:.4f} Ω, increasing Joulean heat loss.\n\n*Action Required*: Decrease charging currents and optimize cooling controls."
            else:
                return "No anomalies detected. The voltage curves, temperature gradients, and resistance measurements are all within nominal bounds."
                
        elif "recommend" in prompt_lower or "action" in prompt_lower or "fix" in prompt_lower:
            return f"Based on the active metrics (SOH: {soh:.1f}%), here are my AI-engineered operational guidelines:\n1. **Charge Capping**: Limit charging current to 0.5C.\n2. **DOD Window**: Avoid discharging below 20% or charging above 80% to lower electrode strain.\n3. **Active Cooling**: Trigger cell fans at 35°C."
            
        else:
            return f"Based on the battery's active telemetry (SOH: {soh:.2f}%, RUL: {int(rul)} cycles, RI: {ri:.4f} Ω), the cell is displaying signs of {health_status.lower()}.\n\nTo preserve remaining capacity, it is advised to avoid deep discharge cycles and high C-rate profiles. Let me know if you would like me to explain specific degradation indicators like Incremental Capacity (ICA) peak shifts or ECM parameter fitting!"

ai_copilot = BatteryAICopilot()

# Load Processed Data
processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processed")
df_features = pd.read_csv(os.path.join(processed_dir, "cycle_features.csv"))
df_metrics = pd.read_csv(os.path.join(processed_dir, "model_comparison_metrics.csv"))
df_predictions = pd.read_csv(os.path.join(processed_dir, "test_predictions.csv"))
df_sim = pd.read_csv(os.path.join(processed_dir, "charging_simulation.csv"))

# Premium Sidebar Navigation Setup
page_mapping = {
    "Overview": "📊 System Overview & Preprocessing",
    "Physics & Features": "⚙️ Feature Engineering & Physics",
    "ML Models": "🤖 Machine Learning Models",
    "BMS & Anomaly": "🚨 Health & Anomaly Center",
    "Smart Charging": "⚡ Intelligent Charging Simulation",
    "Diagnostics Upload": "🔮 Upload & Predict: Custom Diagnostics",
    "Advanced Electrochemistry": "🔋 Advanced Electrochemistry (ICA/ECM)",
    "BMS AI Copilot": "💬 BMS AI Copilot (DeepSeek)"
}

with st.sidebar:
    # Render custom animated charging battery logo
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="battery-glow-icon"></div>
        <h3 style="font-family:'Space Grotesk', sans-serif; font-weight:700; color:#f3f4f6; margin-bottom:2px; font-size:1.4rem;">Battery Health</h3>
        <p style="color:#9ca3af; font-size:0.8rem; margin:0;">NASA Diagnostics Framework</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=list(page_mapping.keys()),
        icons=["speedometer2", "gear", "cpu", "activity", "lightning-charge", "cloud-upload", "battery-half", "robot"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {
                "padding": "5px !important", 
                "background-color": "#111827 !important", 
                "border": "1px solid #1f2937 !important",
                "border-radius": "10px !important"
            },
            "icon": {
                "color": "#1abc9c", 
                "font-size": "16px"
            }, 
            "menu-title": {
                "color": "#9ca3af",
                "font-family": "'Outfit', sans-serif",
                "font-size": "12px",
                "text-transform": "uppercase",
                "letter-spacing": "1px",
                "font-weight": "600",
                "padding-left": "10px"
            },
            "nav-link": {
                "font-family": "'Outfit', sans-serif",
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "0px", 
                "color": "#9ca3af",
                "padding": "10px 15px !important",
                "--hover-color": "#1f2937"
            },
            "nav-link-selected": {
                "background-color": "#1f2937 !important", 
                "color": "#1abc9c !important",
                "font-weight": "600",
                "border-left": "4px solid #1abc9c"
            },
        }
    )
    
    # Global Hugging Face Integration Setup
    st.markdown("---")
    st.subheader("🔑 Hugging Face API")
    default_token = os.getenv("HF_TOKEN", "")
    hf_token = st.text_input("Hugging Face API Token", type="password", value=default_token, help="Enter your Hugging Face API token to query online models.")
    
    selected_hf_model = st.selectbox(
        "Hugging Face Model",
        [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct"
        ],
        index=0
    )

page = page_mapping[selected]

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

            # Generate static report PDF
            pdf_data = generate_pdf_report(
                battery_id="SIM-CELL-01",
                condition="Simulated Static Run",
                cycle=max(1, 200 - sim_rul),
                soh=sim_soh,
                rul=sim_rul,
                health_score=health['score'],
                health_band=health['band'],
                anomaly_status=anomaly['status'],
                mitigation_action=anomaly['mitigation']
            )
            st.download_button(
                label="📥 Download Diagnostic PDF Report",
                data=pdf_data,
                file_name="battery_static_diagnostic_report.pdf",
                mime="application/pdf",
                key="btn_pdf_static"
            )

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
            # Clear old AI reports if new file uploaded
            if "last_uploaded_tab1" not in st.session_state or st.session_state.last_uploaded_tab1 != uploaded_tabular.name:
                st.session_state.last_uploaded_tab1 = uploaded_tabular.name
                if "ai_report_tab1" in st.session_state:
                    del st.session_state.ai_report_tab1
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

                    # Future Prescriptive Recommendations Panel
                    st.markdown("### 📋 Future Prescriptive Recommendations")
                    recs = get_future_recommendations(l_soh, l_rul, latest_row['ri'], latest_row['peak_temp'])
                    
                    rec_cols = st.columns(2)
                    for idx, r in enumerate(recs):
                        col_target = rec_cols[idx % 2]
                        with col_target:
                            col_target.markdown(f"""
                            <div class="card" style="border-left: 4px solid #1abc9c; background-color:#111827; padding: 1rem; margin-bottom: 0.8rem;">
                                <strong style="color:#1abc9c; font-size:1.0rem;">{r['category']}</strong>
                                <div style="color:#f3f4f6; margin-top:0.3rem; font-size:0.95rem;"><b>Recommended Action:</b> {r['action']}</div>
                                <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.25rem;"><i>Expected Impact:</i> {r['impact']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Generate Tabular diagnostic report PDF (with live AI analysis if generated)
                    ai_report_text = st.session_state.get('ai_report_tab1')
                    pdf_up_data = generate_pdf_report(
                        battery_id="CUSTOM-TABULAR",
                        condition="Custom Tabular Upload",
                        cycle=l_cycle,
                        soh=l_soh,
                        rul=l_rul,
                        health_score=calculate_health_score(l_soh, l_rul)['score'],
                        health_band=calculate_health_score(l_soh, l_rul)['band'],
                        anomaly_status=detect_anomaly(l_soh, latest_row['peak_temp'])['status'],
                        mitigation_action=detect_anomaly(l_soh, latest_row['peak_temp'])['mitigation'],
                        ai_analysis=ai_report_text
                    )
                    st.download_button(
                        label="📥 Download Tabular Diagnostics Report (PDF)",
                        data=pdf_up_data,
                        file_name=f"battery_custom_tabular_cycle_{l_cycle}_report.pdf",
                        mime="application/pdf",
                        key="btn_pdf_custom_tab"
                    )
                    
                    # 🤖 AI-Powered Diagnostics Analysis
                    st.markdown("### 🤖 AI-Powered Diagnostics Analysis")
                    if hf_token and hf_token.strip():
                        if st.button("Generate Detailed AI Engineering Report", key="btn_gen_ai_custom_tab1"):
                            prompt = (
                                f"Analyze the following battery diagnostic data from a custom tabular CSV upload:\n"
                                f"- Current Cycle Index: {l_cycle}\n"
                                f"- Predicted SOH: {l_soh:.2f}%\n"
                                f"- Predicted RUL: {int(l_rul)} cycles\n"
                                f"- Internal Resistance: {latest_row['ri']:.4f} Ohms\n"
                                f"- Peak Temperature: {latest_row['peak_temp']:.2f}°C\n"
                                f"- Voltage Drop: {latest_row['voltage_drop']:.3f} V\n\n"
                                f"Provide a detailed, professional battery engineering analysis. Explain what these numbers mean, "
                                f"highlight any critical degradation or thermal safety risks, and recommend specific BMS operational guidelines."
                            )
                            with st.spinner("AI is analyzing the tabular telemetry..."):
                                ai_report = ai_copilot.generate(
                                    prompt, 
                                    context_metrics={"soh": l_soh, "rul": l_rul, "ri": latest_row['ri'], "temp": latest_row['peak_temp']}, 
                                    hf_token=hf_token, 
                                    model_name=selected_hf_model
                                )
                            st.session_state.ai_report_tab1 = ai_report
                            st.rerun()
                        
                        if 'ai_report_tab1' in st.session_state:
                            st.info(st.session_state.ai_report_tab1)
                    else:
                        st.info("💡 To generate AI-powered engineering insights for your uploaded telemetry, please provide a Hugging Face API Token in the sidebar.")
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
            # Clear old AI reports if new file uploaded
            if "last_uploaded_tab2" not in st.session_state or st.session_state.last_uploaded_tab2 != uploaded_raw.name:
                st.session_state.last_uploaded_tab2 = uploaded_raw.name
                if "ai_report_tab2" in st.session_state:
                    del st.session_state.ai_report_tab2
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
                        # Construct 9-channel sequence to match training pipeline
                        t_raw = np.array(time_raw, dtype=np.float32)
                        t_new = np.linspace(t_raw[0], t_raw[-1], 100)
                        
                        v_100 = np.interp(t_new, t_raw, v_raw)
                        curr_100 = np.interp(t_new, t_raw, curr_raw)
                        temp_100 = np.interp(t_new, t_raw, temp_raw)
                        
                        elapsed_time = t_new - t_new[0]
                        dt_new = np.diff(t_new)
                        dt_new = np.insert(dt_new, 0, 0.0)
                        cumulative_capacity = np.cumsum(dt_new * abs(curr_100)) / 3600.0
                        
                        cycle_normalized = np.full(100, custom_cycle_idx / 100.0)
                        ri_channel = np.full(100, ri)
                        mean_rw_temp_channel = np.full(100, 25.0)  # Default environment room temp walk
                        max_rw_temp_channel = np.full(100, 25.0)
                        
                        seq = np.column_stack([
                            v_100, curr_100, temp_100,
                            elapsed_time, cumulative_capacity,
                            cycle_normalized, ri_channel,
                            mean_rw_temp_channel, max_rw_temp_channel
                        ])
                        
                        seq_scaled = (seq - lstm_norm_params['mean']) / lstm_norm_params['std']
                        seq_scaled = np.expand_dims(seq_scaled, axis=0).astype(np.float32)
                        
                        pred_soh_lstm = float(lstm_soh_model.predict(seq_scaled, verbose=0)[0][0] * 100.0)
                        
                        # Guard rails for output predictions
                        pred_soh_lstm = max(0.0, min(100.0, pred_soh_lstm))
                        
                        max_rul_val = lstm_norm_params.get('max_rul', 34.0)
                        pred_rul_lstm = float(lstm_rul_model.predict(seq_scaled, verbose=0)[0][0] * max_rul_val)
                        pred_rul_lstm = max(0.0, pred_rul_lstm)
                        
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

                # Future Prescriptive Recommendations Panel
                st.markdown("### 📋 Future Prescriptive Recommendations")
                target_soh = pred_soh_lstm if lstm_ran else pred_soh_rf
                target_rul = pred_rul_lstm if lstm_ran else pred_rul_rf
                recs = get_future_recommendations(target_soh, target_rul, ri, peak_temp)
                
                rec_cols = st.columns(2)
                for idx, r in enumerate(recs):
                    col_target = rec_cols[idx % 2]
                    with col_target:
                        col_target.markdown(f"""
                        <div class="card" style="border-left: 4px solid #1abc9c; background-color:#111827; padding: 1rem; margin-bottom: 0.8rem;">
                            <strong style="color:#1abc9c; font-size:1.0rem;">{r['category']}</strong>
                            <div style="color:#f3f4f6; margin-top:0.3rem; font-size:0.95rem;"><b>Recommended Action:</b> {r['action']}</div>
                            <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.25rem;"><i>Expected Impact:</i> {r['impact']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Generate raw telemetry report PDF (with live AI analysis if generated)
                ai_report_text_tab2 = st.session_state.get('ai_report_tab2')
                pdf_raw_data = generate_pdf_report(
                    battery_id="CUSTOM-RAW-UNIT",
                    condition="Custom Raw Telemetry Upload",
                    cycle=custom_cycle_idx,
                    soh=pred_soh_rf,
                    rul=pred_rul_rf,
                    health_score=health_state['score'],
                    health_band=health_state['band'],
                    anomaly_status=anomaly_state['status'],
                    mitigation_action=anomaly_state['mitigation'],
                    ai_analysis=ai_report_text_tab2
                )
                st.download_button(
                    label="📥 Download Raw Telemetry Diagnostics Report (PDF)",
                    data=pdf_raw_data,
                    file_name=f"battery_custom_raw_cycle_{custom_cycle_idx}_report.pdf",
                    mime="application/pdf",
                    key="btn_pdf_custom_raw"
                )
                
                # 🤖 AI-Powered Diagnostics Analysis
                st.markdown("### 🤖 AI-Powered Diagnostics Analysis")
                if hf_token and hf_token.strip():
                    if st.button("Generate Detailed AI Engineering Report", key="btn_gen_ai_custom_tab2"):
                        prompt = (
                            f"Analyze the following battery diagnostic data from a custom raw telemetry upload:\n"
                            f"- Current Cycle Index: {custom_cycle_idx}\n"
                            f"- Discharge Capacity: {capacity:.4f} Ah\n"
                            f"- Predicted SOH (Random Forest): {pred_soh_rf:.2f}%\n"
                            f"- Predicted RUL (Random Forest): {int(pred_rul_rf)} cycles\n"
                            f"- Internal Resistance: {ri:.4f} Ohms\n"
                            f"- Peak Temperature: {peak_temp:.2f}°C\n"
                            f"- Voltage Drop: {voltage_drop:#.3f} V\n\n"
                            f"Provide a detailed, professional battery engineering analysis. Explain what these numbers mean, "
                            f"highlight any critical degradation or thermal safety risks, and recommend specific BMS operational guidelines."
                        )
                        with st.spinner("AI is analyzing the raw telemetry..."):
                            ai_report = ai_copilot.generate(
                                prompt, 
                                context_metrics={"soh": pred_soh_rf, "rul": pred_rul_rf, "ri": ri, "temp": peak_temp}, 
                                hf_token=hf_token, 
                                model_name=selected_hf_model
                            )
                        st.session_state.ai_report_tab2 = ai_report
                        st.rerun()
                    
                    if 'ai_report_tab2' in st.session_state:
                        st.info(st.session_state.ai_report_tab2)
                else:
                    st.info("💡 To generate AI-powered engineering insights for your uploaded telemetry, please provide a Hugging Face API Token in the sidebar.")

# ----------------- PAGE 7: ADVANCED ELECTROCHEMISTRY (ICA/ECM) -----------------
elif page == "🔋 Advanced Electrochemistry (ICA/ECM)":
    st.markdown("<h1 class='main-title'>Advanced Electrochemical Diagnostics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Incremental Capacity Analysis (dQ/dV), Differential Voltage Analysis (dV/dQ), and 1-RC Thevenin circuit fitting.</p>", unsafe_allow_html=True)
    
    # Get unique batteries from cycle_features dataframe
    battery_names = sorted(df_features['battery_id'].unique().tolist())
    
    selected_bat = st.selectbox("Select Target Battery", battery_names, key="sel_bat_electro")
    
    if selected_bat:
        # Filter features for selected battery
        df_bat = df_features[df_features['battery_id'] == selected_bat]
        
        tab_ica, tab_ecm, tab_sandbox, tab_lapse = st.tabs([
            "📈 Incremental Capacity (ICA/DVA)",
            "⚡ Equivalent Circuit Fitting (ECM)",
            "🛠️ Virtual Battery Sandbox",
            "⏱️ Battery Aging Time-Lapse"
        ])
        
        # Load base telemetry for scaling
        base_telemetry = pd.read_csv(os.path.join(processed_dir, "sample_telemetry.csv"))
        dt = np.diff(base_telemetry['relativeTime'].values)
        current = base_telemetry['current'].values
        base_capacity = np.zeros(len(base_telemetry))
        for i in range(1, len(base_telemetry)):
            base_capacity[i] = base_capacity[i-1] + abs(current[i-1]) * dt[i-1] / 3600.0
            
        with tab_ica:
            st.write("Incremental Capacity ($dQ/dV$) and Differential Voltage ($dV/dQ$) curves help identify degradation modes like **Loss of Active Material (LAM)** and **Loss of Lithium Inventory (LLI)**.")
            
            available_cycles = sorted(df_bat['cycle'].unique().tolist())
            if len(available_cycles) > 5:
                default_cycles = [available_cycles[0], available_cycles[len(available_cycles)//2], available_cycles[-1]]
            else:
                default_cycles = available_cycles
                
            selected_cycles = st.multiselect("Select Cycles to Compare", available_cycles, default=default_cycles)
            
            if selected_cycles:
                fig_ica = go.Figure()
                fig_dva = go.Figure()
                
                for cyc in sorted(selected_cycles):
                    row = df_bat[df_bat['cycle'] == cyc]
                    if not row.empty:
                        soh_val = row.iloc[0]['soh']
                        ri_val = row.iloc[0]['ri']
                        
                        Q_cyc = base_capacity * (soh_val / 100.0)
                        R_diff = ri_val - 0.0827
                        V_cyc = base_telemetry['voltage'].values - current * R_diff
                        
                        V_mid, dq_dv = calculate_ica(V_cyc, Q_cyc)
                        Q_mid, dv_dq = calculate_dva(V_cyc, Q_cyc)
                        
                        if len(V_mid) > 0:
                            fig_ica.add_trace(go.Scatter(x=V_mid, y=dq_dv, name=f"Cycle {cyc} (SOH {soh_val:.1f}%)", mode='lines', line=dict(width=2)))
                        if len(Q_mid) > 0:
                            fig_dva.add_trace(go.Scatter(x=Q_mid, y=dv_dq, name=f"Cycle {cyc} (SOH {soh_val:.1f}%)", mode='lines', line=dict(width=2)))
                            
                fig_ica.update_layout(
                    title="Incremental Capacity Curves (dQ/dV vs Voltage)",
                    xaxis_title="Voltage (V)",
                    yaxis_title="dQ/dV (Ah/V)",
                    template="plotly_dark",
                    height=450
                )
                st.plotly_chart(fig_ica, use_container_width=True)
                
                fig_dva.update_layout(
                    title="Differential Voltage Curves (dV/dQ vs Capacity)",
                    xaxis_title="Discharge Capacity (Ah)",
                    yaxis_title="dV/dQ (V/Ah)",
                    template="plotly_dark",
                    height=450
                )
                st.plotly_chart(fig_dva, use_container_width=True)
                
        with tab_ecm:
            st.write("Identifies the parameters of a **1-RC Thevenin Equivalent Circuit Model** ($R_0, R_1, C_1$) dynamically as the battery ages.")
            
            cyc_sel = st.selectbox("Select Cycle to Inspect Model Parameters", available_cycles, index=0)
            row_sel = df_bat[df_bat['cycle'] == cyc_sel]
            
            if not row_sel.empty:
                soh_val = row_sel.iloc[0]['soh']
                ri_val = row_sel.iloc[0]['ri']
                
                Q_cyc = base_capacity * (soh_val / 100.0)
                R_diff = ri_val - 0.0827
                V_cyc = base_telemetry['voltage'].values - current * R_diff
                time_cyc = base_telemetry['relativeTime'].values
                
                ecm_params = fit_ecm_parameters(V_cyc, current, time_cyc)
                
                col_ecm1, col_ecm2, col_ecm3, col_ecm4 = st.columns(4)
                with col_ecm1:
                    st.metric("Ohmic Resistance (R₀)", f"{ecm_params['R0']:.4f} Ω")
                with col_ecm2:
                    st.metric("Polarization Resistance (R₁)", f"{ecm_params['R1']:.4f} Ω")
                with col_ecm3:
                    st.metric("Polarization Capacitance (C₁)", f"{ecm_params['C1']:.1f} F")
                with col_ecm4:
                    st.metric("Time Constant (τ)", f"{ecm_params['tau']:.2f} s")
                
                trends_R0 = []
                trends_R1 = []
                trends_C1 = []
                cycles_list = []
                
                for cyc_t in available_cycles[::2]:
                    r_t = df_bat[df_bat['cycle'] == cyc_t]
                    if not r_t.empty:
                        soh_t = r_t.iloc[0]['soh']
                        ri_t = r_t.iloc[0]['ri']
                        Q_t = base_capacity * (soh_t / 100.0)
                        V_t = base_telemetry['voltage'].values - current * (ri_t - 0.0827)
                        params_t = fit_ecm_parameters(V_t, current, time_cyc)
                        
                        cycles_list.append(cyc_t)
                        trends_R0.append(params_t['R0'])
                        trends_R1.append(params_t['R1'])
                        trends_C1.append(params_t['C1'])
                
                st.markdown("### 📈 Equivalent Circuit Model Aging Trends")
                col_gr1, col_gr2 = st.columns(2)
                
                with col_gr1:
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(x=cycles_list, y=trends_R0, name="Ohmic Resistance (R0)", line=dict(color='#e74c3c', width=3)))
                    fig_res.add_trace(go.Scatter(x=cycles_list, y=trends_R1, name="Polarization Resistance (R1)", line=dict(color='#f1c40f', width=3)))
                    fig_res.update_layout(title="Ohmic and Polarization Resistance Fade", xaxis_title="Cycle Index", yaxis_title="Resistance (Ω)", template="plotly_dark", height=350)
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                with col_gr2:
                    fig_cap = go.Figure()
                    fig_cap.add_trace(go.Scatter(x=cycles_list, y=trends_C1, name="Polarization Capacitance (C1)", line=dict(color='#2ecc71', width=3)))
                    fig_cap.update_layout(title="Polarization Capacitance Variation", xaxis_title="Cycle Index", yaxis_title="Capacitance (F)", template="plotly_dark", height=350)
                    st.plotly_chart(fig_cap, use_container_width=True)

        with tab_sandbox:
            st.markdown("### 🛠️ Virtual Battery Sandbox")
            st.write("Design your own electrochemical cell using equivalent circuit parameters and simulate its discharge and thermal response.")
            
            col_sb1, col_sb2 = st.columns([1, 2])
            with col_sb1:
                # Sandbox inputs
                sb_R0 = st.slider("Ohmic Resistance R₀ (Ω)", 0.01, 0.50, 0.08, step=0.01)
                sb_R1 = st.slider("Polarization Resistance R₁ (Ω)", 0.005, 0.30, 0.02, step=0.005)
                sb_C1 = st.slider("Polarization Capacitance C₁ (F)", 100.0, 5000.0, 1500.0, step=100.0)
                sb_cap = st.slider("Nominal Cell Capacity (Ah)", 0.5, 3.0, 1.8, step=0.1)
                sb_I = st.slider("Discharge Current (A)", 0.5, 5.0, 1.5, step=0.1)
                sb_Tamb = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 25.0, step=1.0)
                
            with col_sb2:
                # Simulate CC discharge profile
                t_sim = np.arange(0, 7200, 5) # 5s steps to speed up
                v_sim = []
                temp_sim = []
                soc_sim = []
                v_rc = 0.0
                cell_temp = sb_Tamb
                
                # Thermal constants
                m_cp = 80.0 # Heat capacity J/K
                h_cool = 0.05 # Cooling coefficient W/K
                
                for idx, t_val in enumerate(t_sim):
                    # SOC
                    current_soc = max(0.0, 1.0 - (sb_I * t_val) / (3600.0 * sb_cap))
                    # OCV polynomial
                    v_oc = 3.12 + 1.2 * current_soc - 1.1 * (current_soc**2) + 1.8 * (current_soc**3) - 1.5 * (current_soc**4) + 0.6 * (current_soc**5)
                    
                    # RC voltage update
                    tau_rc = sb_R1 * sb_C1
                    v_rc = sb_I * sb_R1 * (1.0 - np.exp(-t_val / tau_rc))
                    
                    # Cell voltage
                    v_c = v_oc - sb_I * sb_R0 - v_rc
                    
                    # Thermal ODE
                    q_gen = (sb_I**2) * sb_R0 + sb_I * v_rc
                    q_loss = h_cool * (cell_temp - sb_Tamb)
                    dT = (q_gen - q_loss) * 5.0 / m_cp
                    cell_temp += dT
                    
                    if v_c < 2.7 or current_soc <= 0:
                        v_sim.append(v_c)
                        temp_sim.append(cell_temp)
                        soc_sim.append(current_soc)
                        t_sim = t_sim[:idx+1]
                        break
                    
                    v_sim.append(v_c)
                    temp_sim.append(cell_temp)
                    soc_sim.append(current_soc)
                
                # Plot results
                fig_sb = go.Figure()
                fig_sb.add_trace(go.Scatter(x=t_sim, y=v_sim, name="Cell Voltage (V)", line=dict(color="#1abc9c", width=3)))
                fig_sb.update_layout(title="Simulated Discharge Voltage Recovery", xaxis_title="Time (s)", yaxis_title="Voltage (V)", template="plotly_dark", height=280)
                st.plotly_chart(fig_sb, use_container_width=True)
                
                fig_sb_temp = go.Figure()
                fig_sb_temp.add_trace(go.Scatter(x=t_sim, y=temp_sim, name="Temperature (°C)", line=dict(color="#e74c3c", width=3)))
                fig_sb_temp.update_layout(title="Simulated Cell Temperature (Joule Heating)", xaxis_title="Time (s)", yaxis_title="Temperature (°C)", template="plotly_dark", height=280)
                st.plotly_chart(fig_sb_temp, use_container_width=True)

        with tab_lapse:
            st.markdown("### ⏱️ Battery Degradation Time-Lapse")
            st.write("Scrub the slider below to watch how the discharge profiles and incremental capacity peaks morph dynamically over the cell lifetime.")
            
            # Slider
            available_cycles = sorted(df_bat['cycle'].unique().tolist())
            cycle_idx = st.slider("Scrub to Age Cell (Cycle Number)", min_value=int(available_cycles[0]), max_value=int(available_cycles[-1]), value=int(available_cycles[0]))
            
            fig_lapse_v = go.Figure()
            fig_lapse_ica = go.Figure()
            
            bg_cycles = available_cycles[::max(1, len(available_cycles)//8)]
            if cycle_idx not in bg_cycles:
                bg_cycles.append(cycle_idx)
            bg_cycles = sorted(list(set(bg_cycles)))
            
            for cyc in bg_cycles:
                row = df_bat[df_bat['cycle'] == cyc]
                if not row.empty:
                    soh_val = row.iloc[0]['soh']
                    ri_val = row.iloc[0]['ri']
                    
                    Q_cyc = base_capacity * (soh_val / 100.0)
                    R_diff = ri_val - 0.0827
                    V_cyc = base_telemetry['voltage'].values - current * R_diff
                    
                    V_mid, dq_dv = calculate_ica(V_cyc, Q_cyc)
                    
                    if cyc == cycle_idx:
                        color = "#1abc9c"
                        width = 4
                        opacity = 1.0
                        name = f"Current Cycle {cyc}"
                    else:
                        color = "#4a4a4a"
                        width = 1.5
                        opacity = 0.4
                        name = f"Cycle {cyc}"
                        
                    if len(V_cyc) > 0:
                        fig_lapse_v.add_trace(go.Scatter(x=Q_cyc, y=V_cyc, name=name, mode='lines', line=dict(color=color, width=width), opacity=opacity, showlegend=(cyc == cycle_idx)))
                    if len(V_mid) > 0:
                        fig_lapse_ica.add_trace(go.Scatter(x=V_mid, y=dq_dv, name=name, mode='lines', line=dict(color=color, width=width), opacity=opacity, showlegend=(cyc == cycle_idx)))
            
            fig_lapse_v.update_layout(
                title=f"Discharge Voltage Curve (Highlighting Cycle {cycle_idx})",
                xaxis_title="Capacity (Ah)",
                yaxis_title="Voltage (V)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_lapse_v, use_container_width=True)
            
            fig_lapse_ica.update_layout(
                title=f"Incremental Capacity Peaks (Highlighting Cycle {cycle_idx})",
                xaxis_title="Voltage (V)",
                yaxis_title="dQ/dV (Ah/V)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_lapse_ica, use_container_width=True)

# ----------------- PAGE 8: BMS AI COPILOT (DEEPSEEK) -----------------
elif page == "💬 BMS AI Copilot (DeepSeek)":
    st.markdown("<h1 class='main-title'>BMS AI Copilot Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Ask our DeepSeek-powered AI BMS engineer diagnostic questions, recommend operational limits, or inspect anomaly logs.</p>", unsafe_allow_html=True)
    
    if hf_token and hf_token.strip():
        st.success(f"⚡ Hugging Face Inference API Active! Model: `{selected_hf_model}`")
    else:
        # Initialize the DeepSeek model locally
        with st.spinner("Initializing deepseek-ai/DeepSeek-V4-Pro via Transformers..."):
            ai_copilot.init_model()
            
        if ai_copilot.initialized:
            st.success("🤖 DeepSeek-V4-Pro model loaded successfully via Transformers!")
        else:
            st.info("ℹ️ Running in fast lightweight mode (DeepSeek-V4-Pro not loaded). Utilizing high-fidelity context-aware analytical model to simulate DeepSeek responses.")
        
    # Selected cell metrics context for the chatbot
    st.markdown("### 📊 Active Diagnostic Context")
    cells_list = sorted(list(df_features['battery_id'].unique()))
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        selected_cell = st.selectbox("Select Active Cell Context", cells_list)
    with col_c2:
        df_cell = df_features[df_features['battery_id'] == selected_cell].sort_values('cycle')
        max_cycle = int(df_cell['cycle'].max())
        selected_cycle = st.slider("Select Active Cycle", 1, max_cycle, max_cycle)
        
    # Get current cycle metrics
    cell_row = df_cell[df_cell['cycle'] == selected_cycle]
    if not cell_row.empty:
        curr_soh = cell_row.iloc[0]['soh']
        curr_rul = max_cycle - selected_cycle + 25 # Estimate RUL
        curr_ri = cell_row.iloc[0]['ri']
        curr_temp = cell_row.iloc[0]['peak_temp']
    else:
        curr_soh, curr_rul, curr_ri, curr_temp = 85.0, 120, 0.08, 25.0
        
    # Display small visual summary card of the active context
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("SOH", f"{curr_soh:.2f}%")
    with col_m2:
        st.metric("RUL Estimate", f"{int(curr_rul)} cycles")
    with col_m3:
        st.metric("Resistance (RI)", f"{curr_ri:.4f} Ω")
    with col_m4:
        st.metric("Peak Temperature", f"{curr_temp:.1f} °C")
        
    # Chat message log container
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your DeepSeek-powered BMS AI Copilot. How can I help you analyze battery state, thermal limits, or forecast remaining useful life today?"}
        ]
        
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your DeepSeek-powered BMS AI Copilot. How can I help you analyze battery state, thermal limits, or forecast remaining useful life today?"}
        ]
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Chat input
    user_query = st.chat_input("Ask about battery SOH, C-rate capping, thermal anomalies, or physics-guided ML...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Build diagnostic context dictionary
        context = {
            "soh": curr_soh,
            "rul": curr_rul,
            "ri": curr_ri,
            "temp": curr_temp
        }
        
        with st.spinner("DeepSeek is thinking..."):
            ai_response = ai_copilot.generate(
                user_query, 
                context_metrics=context, 
                hf_token=hf_token, 
                model_name=selected_hf_model, 
                chat_history=st.session_state.chat_history
            )
            
        with st.chat_message("assistant"):
            st.write(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
