import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.io import wavfile as sp_wavfile
from tqdm import tqdm
from python_speech_features import mfcc, logfbank
import math


####################createfolder function######################

def ensure_dir(directory):
    """
    Ensure that a directory exists, create it if it does not.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
####################### Plotting functions (modified + new) ##########################


def save_signals(signals, labels, sample_rate, base_output_dir):
    for i, (key, signal) in enumerate(signals.items()):
        label = labels[i]  # 'normal' or 'abnormal'
        output_dir = os.path.join(base_output_dir, label)
        ensure_dir(output_dir)

        time = np.arange(len(signal)) / sample_rate if sample_rate else np.arange(len(signal))
        color = 'green' if label == 'normal' else 'red'

        plt.figure(figsize=(10, 3))
        plt.plot(time, signal, color=color)
        plt.title(f"{key} - Time Domain")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{key}.png")
        plt.savefig(out_path)
        plt.close()






def save_fft(fft_data, labels, output_dir):
    for i, (key, (magnitude, freq)) in enumerate(fft_data.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        plt.figure(figsize=(10, 3))
        plt.plot(freq, magnitude, color='green' if label == 'normal' else 'red')
        plt.title(f"{key} - FFT Magnitude")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()






def save_fbank(fbank_data, labels, output_dir):
    for i, (key, bank) in enumerate(fbank_data.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        img_path = os.path.join(folder, f"{key}.png")
        if os.path.exists(img_path):
            continue  # Skip if image already exists

        cmap = 'Greens' if label == 'normal' else 'Reds'
        plt.figure(figsize=(10, 3))
        plt.imshow(bank, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{key} - Filter Bank")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()






def save_mfcc(mfcc_data, labels, output_dir):
    for i, (key, mfccs) in enumerate(mfcc_data.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        cmap = 'Blues' if label == 'normal' else 'Oranges'
        plt.figure(figsize=(10, 3))
        plt.imshow(mfccs, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{key} - MFCCs")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()



# New: chroma plot grid
def save_chroma(chroma_data, labels, output_dir):
    for i, (key, chroma) in enumerate(chroma_data.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        cmap = 'PuBu' if label == 'normal' else 'YlOrBr'
        plt.figure(figsize=(10, 3))
        plt.imshow(chroma, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{key} - Chroma")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()





def save_tonnetz(tonnetz_data, labels, output_dir):
    for i, (key, tonnetz) in enumerate(tonnetz_data.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        cmap = 'coolwarm' if label == 'normal' else 'cool'
        plt.figure(figsize=(10, 3))
        plt.imshow(tonnetz, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"{key} - Tonnetz")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()



def save_spectral_centroid_bandwidth(signals, labels, sample_rate, output_dir):
    for i, (key, signal) in enumerate(signals.items()):
        label = labels[i]
        folder = os.path.join(output_dir, label)
        ensure_dir(folder)

        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)[0]
        t = librosa.frames_to_time(range(len(spec_cent)), sr=sample_rate)

        plt.figure(figsize=(10, 3))
        plt.plot(t, spec_cent, label='Centroid')
        plt.plot(t, spec_bw, label='Bandwidth', alpha=0.7)
        plt.title(f"{key} - Spectral Centroid & Bandwidth")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{key}.png"))
        plt.close()




# New: waveform with silence highlight for a single signal
def save_waveform_with_silence(signal, label, sr, key, output_dir, top_db=30):
    folder = os.path.join(output_dir, label)
    ensure_dir(folder)

    intervals = librosa.effects.split(signal, top_db=top_db)
    time = np.arange(len(signal)) / sr

    plt.figure(figsize=(14, 3))
    librosa.display.waveshow(signal, sr=sr, alpha=0.6)
    ymin, ymax = plt.ylim()
    last = 0
    for start, end in intervals:
        if last < start:
            plt.axvspan(last/sr, start/sr, color='0.9', alpha=0.6)
        last = end
    if last < len(signal):
        plt.axvspan(last/sr, len(signal)/sr, color='0.9', alpha=0.6)

    plt.title(f"{key} - Silence Highlight")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{key}.png"))
    plt.close()



###################### FFT Calculation Function ##########################
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    y_mag = abs(np.fft.rfft(y) / n)
    return (y_mag, freq)


####################### Main Data Loading and Feature Extraction ####################

audio_root = "./0_dB_fan/fan/id_00"

# Scan folders and build a DataFrame with file paths and labels
filenames = []
labels = []
for label in ['normal', 'abnormal']:
    folder = os.path.join(audio_root, label)

    if not os.path.exists(folder):
        print(f"Warning: folder not found: {folder}")
        continue

    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            file_path = os.path.join(folder, fname)
            filenames.append(file_path)
            labels.append(label)

df = pd.DataFrame({'filename': filenames, 'label': labels})

# Calculate durations
lengths = []
for fpath in df['filename']:
    rate_tmp, signal_tmp = sp_wavfile.read(fpath)
    duration = len(signal_tmp) / rate_tmp
    lengths.append(duration)
df['length'] = lengths

# Class distribution
class_counts = df['label'].value_counts()
fig, ax = plt.subplots()
ax.set_title('Class Distribution by Sample Count', y=1.08)
ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
plt.show()

# Prepare containers
signals = {}
fft_data = {}
fbank_data = {}
mfcc_data = {}
chroma_data = {}
tonnetz_data = {}
sample_rate = None
labels_signals = []

classes = list(np.unique(df['label']))
num_samples_per_class = None  # Number of samples to process per class

for c in classes:
    class_samples = df[df.label == c]
    for i, row in class_samples.iterrows():
        filepath = row['filename']
        print(f"🔄 Loading file: {filepath}")
        signal, rate = librosa.load(filepath, sr=44100)
        print(f"✅ Loaded: {filepath}")
        

        if sample_rate is None:
            sample_rate = rate

        key = f"{c}_{i}"
        print(f"🔬 Extracting features for: {key}")
        signals[key] = signal
        labels_signals.append(c)

        # FFT
        fft_data[key] = calc_fft(signal, rate)

        # Filterbank (log fbank) and MFCC using python_speech_features on first second (or full if shorter)
        take_n = min(len(signal), rate)  # 1 second or full signal if shorter
        try:
            bank = logfbank(signal[:take_n], rate, nfilt=26, nfft=1103).T
            mel = mfcc(signal[:take_n], rate, numcep=13, nfilt=26, nfft=1103).T
        except Exception:
            # Fallback: use librosa-based features if python_speech_features fails
            S = np.abs(librosa.stft(signal[:take_n], n_fft=2048, hop_length=512))**2
            mel_spec = librosa.feature.melspectrogram(S=S, sr=rate, n_mels=26)
            bank = librosa.power_to_db(mel_spec).T
            mel = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13).T

        fbank_data[key] = bank
        mfcc_data[key] = mel

        # Chroma
        chroma = librosa.feature.chroma_stft(y=signal, sr=rate)
        chroma_data[key] = chroma

        # Tonnetz (from harmonic component)
        y_harmonic = librosa.effects.hpss(signal)[0]
        try:
            ton = librosa.feature.tonnetz(y=y_harmonic, sr=rate)
            tonnetz_data[key] = ton
        except Exception:
            # if tonnetz fails (e.g., non-tonal data), store zeros
            tonnetz_data[key] = np.zeros((6, max(1, chroma.shape[1])))

# Create and save visualizations per audio file
output_base = "./visuals"
#print("📈 Saving time-domain plots...")
#save_signals(signals, labels_signals, sample_rate, os.path.join(output_base, "time"))

#print("🔊 Saving FFT plots...")
#save_fft(fft_data, labels_signals, os.path.join(output_base, "fft"))

print("📊 Saving filter bank plots...")
save_fbank(fbank_data, labels_signals, os.path.join(output_base, "fbank"))

print("🎶 Saving MFCC plots...")
save_mfcc(mfcc_data, labels_signals, os.path.join(output_base, "mfcc"))

print("🎵 Saving chroma plots...")
save_chroma(chroma_data, labels_signals, os.path.join(output_base, "chroma"))

print("🎼 Saving tonnetz plots...")
save_tonnetz(tonnetz_data, labels_signals, os.path.join(output_base, "tonnetz"))

print("🔍 Saving spectral centroid and bandwidth plots...")
save_spectral_centroid_bandwidth(signals, labels_signals, sample_rate, os.path.join(output_base, "spec_centroid"))

# Example: detailed visualizations for one normal + one abnormal (if present)
'''normal_key = next((k for k in signals.keys() if k.startswith('normal')), None)
abnormal_key = next((k for k in signals.keys() if k.startswith('abnormal')), None)'''

print("🔊 Saving waveform with silence highlight...")
for i, (key, signal) in enumerate(signals.items()):
    label = labels_signals[i]
    save_waveform_with_silence(signal, label, sample_rate, key, os.path.join(output_base, "silence_highlight"))

'''# Zoomed detailed plots for one sample each
if normal_key:
    pd.Series(signals[normal_key]).plot(figsize=(12, 4), lw=1, title=f'Detailed View: {normal_key}', color='green')
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

if abnormal_key:
    pd.Series(signals[abnormal_key]).plot(figsize=(12, 4), lw=1, title=f'Detailed View: {abnormal_key}', color='red')
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
'''