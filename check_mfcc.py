import numpy as np
import matplotlib.pyplot as plt

# Load the saved files
mfcc_features = np.load("./features/mfcc_features.npy", allow_pickle=True)
mfcc_labels = np.load("./features/mfcc_labels.npy")

# Check number of samples
print("Number of feature samples:", len(mfcc_features))
print("Number of labels:", len(mfcc_labels))

# Check shape of first MFCC sample
print("First MFCC shape:", mfcc_features[0].shape)
print("First label:", mfcc_labels[0])

# Optional: print first few elements
print("\nFirst 3 MFCC feature arrays:")
for i in range(3):
    print(f"Sample {i} shape: {mfcc_features[i].shape}")
    print(mfcc_features[i])
    print("-" * 40)

# Fix for plotting: convert to float array
mfcc_array = np.array(mfcc_features[0], dtype=float)

# Plot the MFCC
plt.imshow(mfcc_array.T, aspect='auto', origin='lower', cmap='viridis')
plt.title("MFCC of Sample 0")
plt.xlabel("Frame")
plt.ylabel("MFCC Coefficients")
plt.colorbar()
plt.tight_layout()
plt.show()
