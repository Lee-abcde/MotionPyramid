import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# ======== Read two npz files ========
npz_file1 = "../../results/difftest124/generate/df10_motion_similarmap_ori.npz"
npz_file2 = "../../results/difftest124/generate/df10_motion.npz"

batch_size = 32

def reshape_to_dim_T_batch(array, batch_size):
    feature_dim, num_frames = array.shape
    num_batches = num_frames // batch_size
    trimmed = array[:, :num_batches * batch_size]  # Drop extra frames
    return trimmed.reshape(feature_dim, batch_size, num_batches, order='F')  # (dim, T, batch)

# ======== Read file 1 ========
data1 = np.load(npz_file1)
positions1 = reshape_to_dim_T_batch(data1['Positions'], batch_size)

# ======== Read file 2 ========
data2 = np.load(npz_file2)
positions2 = reshape_to_dim_T_batch(data2['Positions'], batch_size)

print("File1 positions:", positions1.shape)
print("File2 positions:", positions2.shape)

# ======== Take the sequence from the same batch ========
batch_id = 0
seq1 = positions1[:, batch_id, :].T  # (T, dim)
seq2 = positions2[:, batch_id, :].T  # (T, dim)

assert seq1.shape == seq2.shape, "The two sequences have different lengths or dimensions"

# ======== Compute cross-similarity map ========
# seq1: (T, dim), seq2: (T, dim)
diff = seq1[:, None, :] - seq2[None, :, :]   # (T, T, dim)
dist_matrix = np.linalg.norm(diff, axis=-1)  # (T, T)

# Min-max normalize to [0, 1]
dist_norm = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())

# ======== Visualize ========
plt.imshow(dist_norm, cmap="viridis", origin="lower")
plt.colorbar(label="Normalized L2 distance")
plt.title(f"Cross Similarity Map (Batch {batch_id}, Positions)")
plt.xlabel("Frame index (seq2)")
plt.ylabel("Frame index (seq1)")
plt.show()
