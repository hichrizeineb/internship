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

####################### Plotting functions (modified + new) ##########################

def plot_signals(signals, labels=None, sample_rate=None):
    num_signals = len(signals)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), sharex=False, sharey=True)
    fig.suptitle('Time-Domain Signals (All)', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(signals.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            signal = signals[title]
            time = np.arange(len(signal)) / sample_rate if sample_rate else np.arange(len(signal))
            color = 'blue'
            if labels:
                color = 'green' if labels[i] == 'normal' else 'red'
            axes[x, y].plot(time, signal, color=color)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_fft(fft_data, labels=None):
    num_signals = len(fft_data)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5), sharex=False, sharey=True)
    fig.suptitle('FFT Magnitude Spectra', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(fft_data.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            magnitude, freq = fft_data[title]
            color = 'blue'
            if labels:
                color = 'green' if labels[i] == 'normal' else 'red'
            axes[x, y].plot(freq, magnitude, color=color)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_fbank(fbank_data, labels=None):
    num_signals = len(fbank_data)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5), sharex=False, sharey=True)
    fig.suptitle('Log Filter Bank Energies', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(fbank_data.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            features = fbank_data[title]
            cmap = 'Greens' if labels and labels[i] == 'normal' else 'Reds'
            axes[x, y].imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_mfccs(mfcc_data, labels=None):
    num_signals = len(mfcc_data)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5), sharex=False, sharey=True)
    fig.suptitle('MFCCs', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(mfcc_data.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            features = mfcc_data[title]
            cmap = 'Blues' if labels and labels[i] == 'normal' else 'Oranges'
            axes[x, y].imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# New: chroma plot grid
def plot_chroma(chroma_data, labels=None):
    num_signals = len(chroma_data)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5), sharex=False, sharey=True)
    fig.suptitle('Chroma Features', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(chroma_data.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            features = chroma_data[title]
            cmap = 'PuBu' if labels and labels[i] == 'normal' else 'YlOrBr'
            axes[x, y].imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# New: tonnetz plot grid
def plot_tonnetz(tonnetz_data, labels=None):
    num_signals = len(tonnetz_data)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5), sharex=False, sharey=True)
    fig.suptitle('Tonnetz (Tonal Centroids)', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    i = 0
    keys = list(tonnetz_data.keys())
    for x in range(rows):
        for y in range(cols):
            if i >= num_signals:
                axes[x, y].axis('off')
                continue
            title = keys[i]
            features = tonnetz_data[title]
            cmap = 'coolwarm' if labels and labels[i] == 'normal' else 'cool'
            axes[x, y].imshow(features, aspect='auto', origin='lower', interpolation='nearest', cmap=cmap)
            axes[x, y].set_title(title, fontsize=10)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# New: spectral centroid & bandwidth plot for a list of signals (grid)
def plot_spectral_centroid_bandwidth(signals, sample_rate, labels=None):
    num_signals = len(signals)
    cols = math.ceil(math.sqrt(num_signals))
    rows = math.ceil(num_signals / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*2.5))
    fig.suptitle('Spectral Centroid & Bandwidth', fontsize=20)
    axes = np.array(axes).reshape(rows, cols)

    keys = list(signals.keys())
    for i, key in enumerate(keys):
        x, y = divmod(i, cols)
        sig = signals[key]
        spec_cent = librosa.feature.spectral_centroid(y=sig, sr=sample_rate)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=sig, sr=sample_rate)[0]
        t = librosa.frames_to_time(range(len(spec_cent)), sr=sample_rate)
        axes[x, y].plot(t, spec_cent, label='centroid')
        axes[x, y].plot(t, spec_bw, label='bandwidth', alpha=0.8)
        axes[x, y].set_title(key, fontsize=10)
        axes[x, y].get_xaxis().set_visible(False)
        axes[x, y].get_yaxis().set_visible(False)
        axes[x, y].legend(fontsize=6)
    # hide unused
    for j in range(len(keys), rows*cols):
        x, y = divmod(j, cols)
        axes[x, y].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# New: waveform with silence highlight for a single signal
def plot_waveform_with_silence(signal, sr, title="Waveform with Silence Highlight", top_db=30):
    intervals = librosa.effects.split(signal, top_db=top_db)
    time = np.arange(len(signal)) / sr
    plt.figure(figsize=(14, 3))
    librosa.display.waveshow(signal, sr=sr, alpha=0.6)
    ymin, ymax = plt.ylim()
    # shade silent regions
    last = 0
    for start, end in intervals:
        # region before a voiced interval is silence
        if last < start:
            plt.axvspan(last/sr, start/sr, color='0.9', alpha=0.6)
        last = end
    # region after last voiced interval
    if last < len(signal):
        plt.axvspan(last/sr, len(signal)/sr, color='0.9', alpha=0.6)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.show()


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
num_samples_per_class = 5  # Number of samples to process per class

for c in classes:
    class_samples = df[df.label == c].head(num_samples_per_class)
    for i, row in class_samples.iterrows():
        filepath = row['filename']
        # Use librosa to get floating point mono
        signal, rate = librosa.load(filepath, sr=44100)  # fixed sr for consistency
        if sample_rate is None:
            sample_rate = rate

        key = f"{c}_{i}"
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

# Plot grids (matching the samples we actually created)
plot_signals(signals, labels=labels_signals, sample_rate=sample_rate)
plot_fft(fft_data, labels=labels_signals)
plot_fbank(fbank_data, labels=labels_signals)
plot_mfccs(mfcc_data, labels=labels_signals)
plot_chroma(chroma_data, labels=labels_signals)
plot_tonnetz(tonnetz_data, labels=labels_signals)
plot_spectral_centroid_bandwidth(signals, sample_rate, labels=labels_signals)

# Example: detailed visualizations for one normal + one abnormal (if present)
normal_key = next((k for k in signals.keys() if k.startswith('normal')), None)
abnormal_key = next((k for k in signals.keys() if k.startswith('abnormal')), None)

if normal_key:
    plot_waveform_with_silence(signals[normal_key], sample_rate, title=f"{normal_key} - Waveform with Silence")
if abnormal_key:
    plot_waveform_with_silence(signals[abnormal_key], sample_rate, title=f"{abnormal_key} - Waveform with Silence")

# Zoomed detailed plots for one sample each
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
