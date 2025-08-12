import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram_image(wav_path, save_path):
    """
    Converts a .wav audio file to a mel spectrogram image and saves it.
    """
    try:
        y, sr = librosa.load(wav_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(6, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.axis('off')  # remove axes
        plt.tight_layout(pad=0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"[✓] Saved image: {save_path}")
    except Exception as e:
        print(f"[✗] Failed for {wav_path}: {e}")


def convert_dataset_to_images(audio_root, output_root):
    """
    Traverses 'normal' and 'abnormal' folders under audio_root, 
    and saves spectrogram images under output_root.
    """
    for label in ["normal", "abnormal"]:
        folder = os.path.join(audio_root, label)
        folder = os.path.normpath(folder) 
        if not os.path.isdir(folder):
            print(f"[!] Skipping: {folder} (not found)")
            continue

        for fname in os.listdir(folder):
            if not fname.endswith(".wav"):
                continue

            wav_path = os.path.join(folder, fname)

            # Replace audio root with image output root and .wav → .png
            save_path = os.path.join(
                output_root,
                label,
                fname.replace(".wav", ".png")
            )

            create_spectrogram_image(wav_path, save_path)


if __name__ == "__main__":
    # CHANGE these paths as needed
    audio_path = "./0_dB_fan/fan/id_00"       # original dataset location
    image_output_path = "./images/fan/id_00" # output folder for images

    convert_dataset_to_images(audio_path, image_output_path)
