import numpy as np

# Load the labels
labels = np.load("features/mfcc_labels.npy")

print(f"✅ Total Labels: {len(labels)}")

# Print unique labels and their counts
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    label_name = "Normal" if u == 0 else "Abnormal"
    print(f"{label_name} ({u}): {c} samples")
print("\n✅ Label check complete.")