# train_pipeline.py
import numpy as np
import mne
from mne.datasets import eegbci
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

print("==================================================")
print("🧠 BCI BACKEND PIPELINE INITIALIZATION")
print("==================================================")

# Step 1: Load EEG Dataset securely using the correct argument format
print("\n[1/6] Downloading & loading PhysioNet Motor Imagery EEG Data...")
# Fetch Subject 1, Run 4 (Imagined left vs right hand) and Run 6 (Imagined hand vs foot)
files = eegbci.load_data(subjects=1, runs=[4, 6])
raw_files = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
raw = mne.concatenate_raws(raw_files)

# Step 2: Temporal Filtering (Remove high-frequency muscle noise and low-frequency DC offset)
print("\n[2/6] Applying Bandpass Filter (1 Hz - 40 Hz)...")
raw.filter(1., 40., fir_design='firwin', verbose=False)

# Step 3: Event Extraction and Epoch Slicing
print("\n[3/6] Slicing signal stream into task-bound Epochs...")
events, event_id = mne.events_from_annotations(raw, verbose=False)
picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')

# Extract baseline temporal windows from 1.0s to 3.5s post-stimulus target cue
epochs = mne.Epochs(raw, events, event_id=None, tmin=1.0, tmax=3.5, 
                    proj=True, picks=picks, baseline=None, preload=True, verbose=False)

X_raw = epochs.get_data(copy=True)  # Matrix array shape: (n_epochs, n_channels, n_times)
y_raw = epochs.events[:, 2]

# Step 4: Spatial-Temporal Feature Extraction
print("\n[4/6] Extracting Signal Features (Mean Absolute Value & Variance)...")
def compute_bci_features(data_matrix):
    # Calculate statistical summaries along the temporal data axis
    mav = np.mean(np.abs(data_matrix), axis=-1)
    variance = np.var(data_matrix, axis=-1)
    return np.hstack((mav, variance))

X_features = compute_bci_features(X_raw)

# Step 5: Train Classifier
print("\n[5/6] Partitioning datasets & training Linear Support Vector Machine (SVM)...")
X_train, X_test, y_train, y_test = train_test_split(X_features, y_raw, test_size=0.3, random_state=42)

model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Classifier Training Successful! Validation Accuracy: {accuracy * 100:.2f}%")

# Step 6: Serialization
print("\n[6/6] Serializing and storing model components...")
joblib.dump(model, 'bci_svm_model.pkl')
joblib.dump(X_raw, 'sample_eeg_signal.pkl')
print("📦 Saved 'bci_svm_model.pkl' and 'sample_eeg_signal.pkl' to working directory!")
print("\nBackend execution complete. Proceed to run your frontend UI pipeline.")
