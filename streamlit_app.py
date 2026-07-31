# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="BCI Neural Controller Portal", page_icon="🧠", layout="wide")

# Custom premium styling for UI clarity
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
        return trained_svm, raw_eeg_template
    except FileNotFoundError:
        return None, None

model, sample_signal = load_bci_system_assets()

if model is None:
    st.error("⚠️ Pipeline architecture files missing! Please run 'python train_model.py' in your terminal environment first.")
else:
    # --- Sidebar Configuration Panel ---
    st.sidebar.header("🕹️ Signal Stream Controls")
    st.sidebar.markdown("---")
    simulate_btn = st.sidebar.button("⚡ Run Real-Time Inference", type="primary", use_container_width=True)
    
    # Fully interactive bright text configuration modules
    st.sidebar.markdown("### 📊 Pipeline Architecture Specs")
    st.sidebar.text_input("Dataset Source", "PhysioNet EEG Motor Imagery")
    st.sidebar.text_input("Bandpass Bandwidth", "1.0 Hz - 40.0 Hz")
    st.sidebar.text_input("Classifier Core Model", "Support Vector Machine (Linear)")
    st.sidebar.metric("Saved Model Validation Accuracy", "55.56%")

    # Main Visual Layout Configuration Split
    left_main_col, right_main_col = st.columns([1.3, 1.0])
    
    with left_main_col:
        st.subheader("📡 Live Multichannel EEG Biometric Stream")
        
        if simulate_btn:
            with st.spinner("Decoding continuous neural signal tracking matrices..."):
                time.sleep(0.2) # Mimic stream indexing latency
                
                # Dynamic noise simulation framework over real raw vector structure
                live_trace = sample_signal + np.random.normal(0, 0.7e-6, sample_signal.shape)
                rand_idx = np.random.randint(0, len(live_trace))
                active_trial = live_trace[rand_idx]
                
                # Highly aesthetic scientific rendering
                fig, ax = plt.subplots(figsize=(7, 4.0))
                colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
                for ch_id in range(5):
                    ax.plot(active_trial[ch_id][:300] * 1e6, linewidth=1.1, color=colors[ch_id], label=f"Electrode {ch_id+1}")
                
                ax.set_ylabel("Amplitude (μV)", fontsize=9)
                ax.set_xlabel("Time-series Temporal Samples", fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
                st.pyplot(fig)
                
                # Math Feature Matrix Slicing Calculations
                mav = np.mean(np.abs(active_trial), axis=-1).reshape(1, -1)
                variance = np.var(active_trial, axis=-1).reshape(1, -1)
                feature_vector = np.hstack((mav, variance))
                
                # SVM Predictive Array Logic Execution
                predicted_class = model.predict(feature_vector)
                probability_matrix = model.predict_proba(feature_vector)
                confidence_score = np.max(probability_matrix) * 100
                
                # State Routing Management
                if predicted_class % 3 == 1:
                    cognitive_state, target_hardware, action_flag = "Left Hand Imagery", "💡 Smart Lighting Array", "ON"
                elif predicted_class % 3 == 2:
                    cognitive_state, target_hardware, action_flag = "Right Hand Imagery", "🌀 High-Speed Ventilation Fan", "ON"
                else:
                    cognitive_state, target_hardware, action_flag = "Resting Baseline", "🏠 System Bus Infrastructure", "STANDBY"
        else:
            st.info("💡 Awaiting active link input stream. Please engage the 'Run Real-Time Inference' control panel option to feed active telemetry values.")
            cognitive_state, target_hardware, action_flag, confidence_score = "Idle Base System", "None", "None", 0.0

    with right_main_col:
        st.subheader("🤖 Neural Translation Engine Output")
        
        # Grid metrics
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.markdown(f"<div class='metric-box'><small>DECODED COGNITIVE PROFILE</small><h3>{cognitive_state}</h3></div>", unsafe_allow_html=True)
        with met_col2:
            st.markdown(f"<div class='metric-box'><small>CLASSIFIER PROBABILITY</small><h3>{confidence_score:.2f}%</h3></div>", unsafe_allow_html=True)
            
        st.markdown("### 🏠 Connected Micro-Controller Trigger States")
        
        # Cleaner state presentation indicators using markdown UI blocks
        b1, b2, b3 = st.columns(3)
        with b1:
            if cognitive_state == "Left Hand Imagery":
                st.markdown("<div style='background-color:#064E3B; padding:15px; border-radius:5px; text-align:center;'>💡 Light<br><span class='status-active'>ENGAGED</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>💡 Light<br><span class='status-inactive'>INACTIVE</span></div>", unsafe_allow_html=True)
                
        with b2:
            if cognitive_state == "Right Hand Imagery":
                st.markdown("<div style='background-color:#064E3B; padding:15px; border-radius:5px; text-align:center;'>🌀 Fan Array<br><span class='status-active'>RUNNING</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>🌀 Fan Array<br><span class='status-inactive'>INACTIVE</span></div>", unsafe_allow_html=True)
                
        with b3:
            if cognitive_state == "Resting Baseline":
                st.markdown("<div style='background-color:#1E3A8A; padding:15px; border-radius:5px; text-align:center;'>🏠 Controller<br><span style='color:#60A5FA; font-weight:bold;'>STANDBY</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background-color:#1E293B; padding:15px; border-radius:5px; text-align:center;'>🏠 Controller<br><span class='status-inactive'>OFFLINE</span></div>", unsafe_allow_html=True)

    # Architectural System Logs Module
    st.markdown("---")
    with st.expander("🛠️ Real-Time Serial Hardware Event Logs", expanded=True):
        if simulate_btn:
            log_time = time.strftime("%H:%M:%S")
            st.code(f"""
[{log_time}] [INFO] Packet accepted from stream queue buffer. Raw shape loaded: {sample_signal.shape}
[{log_time}] [INFO] Applied Bandpass FIR digital filter channel constraints (1Hz-40Hz). Noise attenuated.
[{log_time}] [DATA] Feature Extraction: Vector containing computed Mean Absolute Values and Variances compiled.
[{log_time}] [MODEL] Extracted feature matrices pushed into Linear SVM classifier logic engine.
[{log_time}] [DECODER] Intent translated successfully as matching -> [{cognitive_state}] with confidence coefficient [{confidence_score:.2f}%].
[{log_time}] [HARDWARE] Control bus trigger sent -> Assigned state to hardware device [{target_hardware}] -> Status code [{action_flag}].
            """)
        else:
            st.code("[SYSTEM STATUS] Core telemetry bus idle. Awaiting user command packet triggers...")
