# 🧠 Brain-Computer Interface (BCI) Smart Home System

An end-to-end signal processing and machine learning pipeline that translates human EEG motor imagery signals into real-time smart home device triggers.

## 🚀 System Architecture
1. **Data Stream**: PhysioNet EEG Motor Imagery Dataset (via MNE-Python).
2. **Preprocessing**: 1-40 Hz FIR Bandpass Filter to isolate Alpha/Beta bands.
3. **Feature Extraction**: Mean Absolute Value (MAV) & Signal Variance.
4. **Classification**: Linear Support Vector Machine (SVM).
5. **Frontend Application**: Streamlit web dashboard.
## 📦 Installation & Setup
```bash
pip install mne scikit-learn streamlit matplotlib joblib
python train_pipeline.py
streamlit run streamlit_app.py
```
