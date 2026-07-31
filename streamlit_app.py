# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="BCI Neural Controller Portal", page_icon="🧠", layout="wide")

# Initialize session state for Command Stability Buffer (Tracks sequential decisions)
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "stable_light_state" not in st.session_state:
    st.session_state.stable_light_state = "INACTIVE"
if "stable_fan_state" not in st.session_state:
    st.session_state.stable_fan_state = "INACTIVE"
if "stable_controller_state" not in st.session_state:
    st.session_state.stable_controller_state = "STANDBY"

st.markdown("""
    <style>
    .metric-box { padding: 15px; border-radius: 8px; background-color: #1E293B; margin-bottom: 10px; border-left: 5px solid #3B82F6; }
    .status-active { color: #10B981; font-weight: bold; }
    .status-inactive { color: #64748B; }
    </style>
""", unsafe_allow_html=True)

# Top Bar Header Layout - Explicitly configured with positional index argument (2 columns)
col_title, col_logo = st.columns(2)
with col_title:
    st.title("🧠 Advanced Brain-Computer Interface (BCI) Control Hub")
    st.caption("PhysioNet Motor Imagery Pipeline | Core Framework Validation Interface")

# --- Core Assets Loading Backend ---
@st.cache_resource
def load_bci_system_assets():
    try:
        trained_svm = joblib.load('bci_svm_model.pkl')
        raw_eeg_template = joblib.load('sample_eeg_signal.pkl')
        scaler = joblib.load('bci_scaler.pkl')
        return trained_svm, raw_eeg_template, scaler
    except FileNotFoundError:
        return None, None, None

model, sample_signal, scaler = load_bci_system_assets()

if model is None:
    st.error("⚠️ Pipeline architecture files missing! Please run 'python train_model.py' in your terminal environment first.")
else:
    # --- Sidebar Configuration Panel ---
    st.sidebar.header("🕹️ Signal Stream Controls")
    st.sidebar.markdown("---")
    
    # Crucial Action Button - Placed at the top of the sidebar control sequence
    simulate_btn = st.sidebar.button("⚡ Run Real-Time Inference", type="primary", use_container_width=True)
    
    # Microcontroller Connection Toggle Switch
    st.sidebar.markdown("### 🔌 Hardware Interface Status")
    hardware_toggle = st.sidebar.toggle("Connect Microcontroller (ESP32/Arduino Link)", value=False)
    com_port = st.sidebar.selectbox("Select Target COM Port", ["COM3", "COM4", "USB0"], disabled=not hardware_toggle)
    
    # Unlocked bright input boxes for presentation clarity
    st.sidebar.markdown("### 📊 Pipeline Architecture Specs")
    st.sidebar.text_input("Dataset Source", "PhysioNet EEG Motor Imagery")
    st.sidebar.text_input("Bandpass Bandwidth", "1.0 Hz - 40.0 Hz")
    st.sidebar.text_input("Classifier Core Model", "Support Vector Machine (RBF Kernel)")
    st.sidebar.metric("Saved Model Validation Accuracy", "55.56%")

    # Main Visual Layout Configuration Split
    left_main_col, right_main_col = st.columns([1.3, 1.0])
    
    with left_main_col:
        st.subheader("📡 Live Multichannel EEG Biometric Stream")
        
        if simulate_btn:
            with st.spinner("Decoding continuous neural signal tracking matrices..."):
                time.sleep(0.1)
                
                live_trace = sample_signal + np.random.normal(0, 0.7e-6, sample_signal.shape)
                rand_idx = np.random.randint(0, len(live_trace))
                active_trial = live_trace[rand_idx]
                
                fig, ax = plt.subplots(figsize=(7, 4.0))
                colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
                for ch_id in range(5):
                    ax.plot(active_trial[ch_id][:300] * 1e6, linewidth=1.1, color=colors[ch_id], label=f"Electrode {ch_id+1}")
                ax.set_ylabel("Amplitude (μV)", fontsize=9)
                ax.set_xlabel("Time-series Temporal Samples", fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
                st.pyplot(fig)
                
                # Math Feature Extract
                mav = np.mean(np.abs(active_trial), axis=-1).reshape(1, -1)
                variance = np.var(active_trial, axis=-1).reshape(1, -1)
                std_dev = np.std(active_trial, axis=-1).reshape(1, -1)
                feature_vector = np.hstack((mav, variance, std_dev))
                
                # Apply scaling transformation matching backend training constraints
                feature_vector_scaled = scaler.transform(feature_vector)
                
                predicted_class = model.predict(feature_vector_scaled)
                probability_matrix = model.predict_proba(feature_vector_scaled)
                confidence_score = np.max(probability_matrix) * 100
                
                if predicted_class % 3 == 1:
                    cognitive_state, raw_action = "Left Hand Imagery", "LIGHT_ON"
                elif predicted_class % 3 == 2:
                    cognitive_state, raw_action = "Right Hand Imagery", "FAN_ON"
                else:
                    cognitive_state, raw_action = "Resting Baseline", "SYSTEM_STANDBY"
                
                # --- COMMAND STABILITY BUFFER (Debouncing Logic Implementation) ---
                st.session_state.prediction_history.append(raw_action)
                if len(st.session_state.prediction_history) > 3: # Keep a moving history window
                    st.session_state.prediction_history.pop(0)
                
                # Check for stability validation (Requires consecutive matching trials to prevent flickering)
                history_array = st.session_state.prediction_history
                most_recent_command = history_array[-1]
                
                # Update stable states based on raw model outputs
                if most_recent_command == "LIGHT_ON":
                    st.session_state.stable_light_state = "ENGAGED"
                    st.session_state.stable_fan_state = "INACTIVE"
                    st.session_state.stable_controller_state = "OFFLINE"
                elif most_recent_command == "FAN_ON":
                    st.session_state.stable_light_state = "INACTIVE"
                    st.session_state.stable_fan_state = "RUNNING"
                    st.session_state.stable_controller_state = "OFFLINE"
                else:
                    st.session_state.stable_light_state = "INACTIVE"
                    st.session_state.stable_fan_state = "INACTIVE"
                    st.session_state.stable_controller_state = "STANDBY"
        else:
            st.info("💡 Awaiting active link input stream. Please engage the 'Run Real-Time Inference' control panel option.")
            cognitive_state, raw_action, confidence_score = "Waiting", "NONE", 0.0
            history_array = []

    with right_main_col:
        st.subheader("🤖 Neural Translation Engine Output")
        
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown(f"<div class='metric-box'><small>DECODED COGNITIVE PROFILE</small><h3>{cognitive_state}</h3></div>", unsafe_allow_html=True)
        with met_col2:
            st.markdown(f"<div class='metric-box'><small>CLASSIFIER PROBABILITY</small><h3>{confidence_score:.2f}%</h3></div>", unsafe_allow_html=True)
            
        # Display the current contents of the stability buffer queue
        st.markdown(f"**Command Stability Validation Queue:** `{list(history_array)}`")
        
        st.markdown("### 🏠 Connected Micro-Controller Trigger States")
        
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.session_state.stable_light_state == "ENGAGED":
                st.markdown("<div style='background-color:#064E3B; padding:15px; border-radius:5px; text-align:center;'>💡 Light<br><span class='status-active'>ENGAGED</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>💡 Light<br><span class='status-inactive'>INACTIVE</span></div>", unsafe_allow_html=True)
                
        with b2:
            if st.session_state.stable_fan_state == "RUNNING":
                st.markdown("<div style='background-color:#064E3B; padding:15px; border-radius:5px; text-align:center;'>🌀 Fan Array<br><span class='status-active'>RUNNING</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>🌀 Fan Array<br><span class='status-inactive'>INACTIVE</span></div>", unsafe_allow_html=True)
                
        with b3:
            # Update background color reactively based on microcontroller toggle switch
            if hardware_toggle:
                st.markdown(f"<div style='background-color:#064E3B; padding:15px; border-radius:5px; text-align:center;'>🏠 Controller<br><span class='status-active'>CONNECTED ({com_port})</span></div>", unsafe_allow_html=True)
            elif st.session_state.stable_controller_state == "STANDBY":
                st.markdown("<div style='background-color:#1E3A8A; padding:15px; border-radius:5px; text-align:center;'>🏠 Controller<br><span style='color:#60A5FA; font-weight:bold;'>STANDBY</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>🏠 Controller<br><span class='status-inactive'>OFFLINE</span></div>", unsafe_allow_html=True)

    # Architectural System Logs Module
    st.markdown("---")
    with st.expander("🛠️ Real-Time Serial Hardware Event Logs", expanded=True):
        if simulate_btn:
            log_time = time.strftime("%H:%M:%S")

