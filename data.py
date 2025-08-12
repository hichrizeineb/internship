import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from python_speech_features import mfcc, logfbank

# ========== CONFIG ==========
AUDIO_ROOT = "./0_dB_fan/fan/id_00"   # Adjust this path if needed
SAMPLE_RATE = 16000  # Match dataset specs
CHUNK_DURATION = 1.0  # seconds
N_MFCC = 13
N_FILT = 26
N_FFT = 1024  # FFT size for feature extraction

# ========== UTILS ==========
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ========== STEP 1: Load paths ==========
filenames = []
labels = []
for label in ["normal", "abnormal"]:
    folder = os.path.join(AUDIO_ROOT, label)
    if not os.path.exists(folder):
        print(f"Warning: folder not found: {folder}")
        continue

    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            file_path = os.path.join(folder, fname)
            filenames.append(file_path)
            labels.append(label)

df = pd.DataFrame({"filename": filenames, "label": labels})

# ========== STEP 2: Feature Extraction ==========
mfcc_features = []
mfcc_labels = []

print("🔍 Extracting MFCC features in 1s chunks...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    filepath = row['filename']
    label = row['label']

    signal, rate = librosa.load(filepath, sr=SAMPLE_RATE)
    total_len = len(signal)
    samples_per_chunk = int(CHUNK_DURATION * SAMPLE_RATE)
    num_chunks = total_len // samples_per_chunk

    for c in range(num_chunks):
        chunk = signal[c * samples_per_chunk:(c + 1) * samples_per_chunk]

        # Using python_speech_features for MFCC
        try:
            mfcc_feat = mfcc(chunk, samplerate=SAMPLE_RATE, numcep=N_MFCC, nfilt=N_FILT, nfft=N_FFT)
        except Exception as e:
            print(f"Error processing {filepath} chunk {c}: {e}")
            continue

        mfcc_features.append(mfcc_feat)  # shape: (frames, N_MFCC)
        mfcc_labels.append(0 if label == 'normal' else 1)  # Binary label

# ========== STEP 3: Save ==========
print(f"\n✅ Extracted {len(mfcc_features)} feature samples")

ensure_dir("./features")
np.save("./features/mfcc_features.npy", np.array(mfcc_features, dtype=object))  # Ragged arrays
np.save("./features/mfcc_labels.npy", np.array(mfcc_labels))
print("💾 Features saved to ./features/")
