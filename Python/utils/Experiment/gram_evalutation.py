import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import itertools
import math
import matplotlib
matplotlib.use("TkAgg")
def extract_fragment(X, b, t, window):
    """
    Extract from X (D, B, T) the fragment for style b starting at phase t with length window, using wrap-around
    Return fragment shape: (D, window)
    """
    D, B, T = X.shape
    idxs = [(t + i) % T for i in range(window)]
    return X[:, b, idxs]  # (D, window)

def feature_avg(fragment):
    """
    Average-pooled feature: average over the time dimension and return a 1D vector (D,)
    """
    return fragment.mean(axis=1)  # (D,)

def feature_gram(fragment, use_norm=True):
    """
    Gram feature: fragment shape (D, window)
    Compute G = (fragment @ fragment.T) / window -> (D, D)
    Vectorize the upper triangle including the diagonal, L2-normalize it, and return vector v
    If D^2 is too large, apply PCA or dimensionality reduction externally; this implementation stays direct
    """
    D, W = fragment.shape
    G = (fragment @ fragment.T) / float(W)  # (D, D)
    # vectorize upper triangle
    idxs = np.triu_indices(D)
    v = G[idxs]
    if use_norm:
        n = np.linalg.norm(v) + 1e-12
        v = v / n
    return v

def mean_pairwise_distance(features, metric='euclidean'):
    """
    features: list/array of vectors (n_samples, dim) or [v1, v2, ...]
    Return the average pairwise distance over all samples (mean pairwise distance)
    metric currently only supports 'euclidean' or 'cosine'(cosine distance = 1 - cosine_sim)
    """
    if isinstance(features, list):
        arr = np.vstack(features)
    else:
        arr = np.asarray(features)
    n = arr.shape[0]
    if n <= 1:
        return 0.0
    dsum = 0.0
    count = 0
    if metric == 'euclidean':
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(arr[i] - arr[j])
                dsum += d
                count += 1
    elif metric == 'cosine':
        # cosine distance = 1 - cosine_similarity
        from sklearn.metrics.pairwise import cosine_similarity
        S = cosine_similarity(arr)
        # upper triangle excluding diagonal
        for i in range(n):
            for j in range(i+1, n):
                dsum += (1.0 - S[i, j])
                count += 1
    else:
        raise ValueError("Unsupported metric")
    return dsum / max(1, count)

def compute_phase_robustness(X, window=4, metric='euclidean', use_norm_for_gram=True):
    """
    Compute phase robustness for each style:
      - X: (D, B, T)
      - window: number of frames around each phase to use as fragment length; recommended 2 to 8
      - metric: 'euclidean' or 'cosine'
    Returns:
      results = {
        'avg_scores': np.array(shape=(B,)),  # mean pairwise distance for each style (avg feature)
        'gram_scores': np.array(shape=(B,)),
      }
    Lower values mean the same action is more stable across phases and therefore more robust.
    """
    D, B, T = X.shape
    avg_scores = []
    gram_scores = []

    for b in range(B):
        # For style b, compute fragment features for all start phases
        avg_feats = []
        gram_feats = []
        for t in range(T):
            frag = extract_fragment(X, b, t, window)  # (D, window)
            avg_feats.append(feature_avg(frag))       # (D,)
            gram_feats.append(feature_gram(frag, use_norm=use_norm_for_gram))  # (D*(D+1)/2,)

        # Compute mean pairwise distance for this style across phases
        avg_score = mean_pairwise_distance(avg_feats, metric=metric)
        gram_score = mean_pairwise_distance(gram_feats, metric=metric)

        avg_scores.append(avg_score)
        gram_scores.append(gram_score)

    avg_scores = np.array(avg_scores)
    gram_scores = np.array(gram_scores)
    return {'avg_scores': avg_scores, 'gram_scores': gram_scores}

def summarize_and_plot(results, metric_name='Euclidean distance'):
    """
    Print summary, run paired t-test, and draw box plots
    """
    avg_scores = results['avg_scores']
    gram_scores = results['gram_scores']
    B = len(avg_scores)

    print("Per-style mean pairwise distances:")
    print(f"Average pooling: mean = {avg_scores.mean():.6f}, std = {avg_scores.std(ddof=1):.6f}")
    print(f"Gram matrix    : mean = {gram_scores.mean():.6f}, std = {gram_scores.std(ddof=1):.6f}")

    # paired t-test
    tstat, pval = ttest_rel(avg_scores, gram_scores)
    print(f"Paired t-test (avg vs gram): t = {tstat:.4f}, p = {pval:.4e}")
    if pval < 0.05:
        print("=> Difference is statistically significant (p < 0.05).")
    else:
        print("=> No statistically significant difference (p >= 0.05).")

    # boxplot
    plt.figure(figsize=(6,4))
    data = [avg_scores, gram_scores]
    plt.boxplot(data, labels=['Average', 'Gram'])
    plt.ylabel(metric_name)
    plt.title("Phase robustness: mean pairwise distance per style")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Example usage: read npz and run
# -----------------------------
def run_phase_robustness_on_npz(npz_file, key='Positions', batch_size=64, window=4):
    """
    Compatible with your original reshape_to_dim_T_batch interface:
      - npz_file: path
      - key: key to read from npz: Positions, Velocities, or Rotations
      - batch_size: your previous batch_size corresponds to B; note the reshape convention
    return result dictionary
    """
    data = np.load(npz_file)
    arr = data[key]  # assuming the original shape is (D, total_frames)
    # your previous reshape_to_dim_T_batch converted (D, total_frames) -> (D, batch_size, num_batches)
    D, total_frames = arr.shape
    num_batches = total_frames // batch_size
    arr_reshaped = arr.reshape(D, batch_size, num_batches, order='F')  # (D, B, T)
    results = compute_phase_robustness(arr_reshaped, window=window, metric='euclidean', use_norm_for_gram=True)
    summarize_and_plot(results, metric_name='Mean pairwise Euclidean distance')
    return results

# Run example; change to your actual path
if __name__ == "__main__":
    npz_file = "../../results/difftest116/generate/gt2_motion.npz"  # change to your path
    res = run_phase_robustness_on_npz(npz_file, key='Positions', batch_size=32, window=64)
