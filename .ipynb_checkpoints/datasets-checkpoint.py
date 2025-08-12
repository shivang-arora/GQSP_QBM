import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def load_and_preprocess_mnist(classes=[0,1,2,3], n_components=4, random_state=42):
    """
    1) Fetch MNIST, 
    2) Filter to `classes`, 
    3) PCA→n_components, 
    4) Normalize each PCA feature to zero-mean/unit-variance.
    
    Returns:
      X_norm : (n_samples, n_components)  normalized PCA features
      y_sel  : (n_samples,)               labels
      X_raw  : (n_samples, 784)           raw pixel data
      scaler : fitted StandardScaler      (in case you want to transform new data)
      pca    : fitted PCA                 (for inverse_transform or future batches)
    """
    # fetch full MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_raw, y = mnist['data'], mnist['target'].astype(int)
    
    X_raw=X_raw[0:1000][:]
    y=y[0:1000]
    # filter to the desired classes
    mask = np.isin(y, classes)
    X_sel, y_sel = X_raw[mask], y[mask]

    # PCA reduction
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_sel)

    # normalization
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pca)

    return X_norm, y_sel, X_sel, pca, scaler

def visualize_samples(X_raw, y, sample_indices, n_cols=5):
    """
    Plots the 28×28 images at positions `sample_indices` from X_raw.
    """
    
    n = len(sample_indices)
    n_rows = (n + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols*2, n_rows*2))
    for i, idx in enumerate(sample_indices):
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.imshow(X_raw[idx].reshape(28,28), cmap='gray')
        ax.set_title(f"Label {y[idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def to_binary(y):
    classes=np.max(y)
   
    n = int(classes).bit_length()
    bin_y=[]
    for k in y:
        binary=[]
        val=format(k, '0'+str(n)+'b')
        for i in val:
            
            binary.append(int(i))
        bin_y.append(binary)
    return np.array(bin_y)

def make_nonlinear_classification(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    blobs_per_class=2,
    separability=1.0,        # higher => easier (centers farther, blobs tighter)
    ensure_nonlinear=True,
    random_state=42,
    max_retries=3,
):
    rng = np.random.RandomState(random_state)

    # --- 1) Layout: total blobs and centers ---
    if isinstance(blobs_per_class, int):
        bpc_list = [blobs_per_class] * n_classes
    else:
        assert len(blobs_per_class) == n_classes
        bpc_list = blobs_per_class
    total_blobs = sum(bpc_list)

    # place centers on a sphere and scale outward with separability
    centers = rng.normal(size=(total_blobs, n_features))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= (3.0 * separability + 1.0)

    # samples per blob (balanced overall)
    base = n_samples // total_blobs
    counts = np.full(total_blobs, base, dtype=int)
    counts[: n_samples - base * total_blobs] += 1

    # blob spread inversely w.r.t. separability
    cluster_std = max(0.10, 1.5 / (separability + 1e-6))

    # --- 2) Draw blobs, assign class by blob index ---
    X_parts, blob_ids = [], []
    for i, c in enumerate(counts):
        Xi, _ = make_blobs(
            n_samples=c,
            n_features=n_features,
            centers=centers[i : i + 1],
            cluster_std=cluster_std,
            random_state=rng.randint(0, 10_000),
        )
        X_parts.append(Xi)
        blob_ids.append(np.full(c, i))
    X = np.vstack(X_parts)
    blob_ids = np.concatenate(blob_ids)

    # map blobs -> classes
    blob_to_class = []
    for cls, bpc in enumerate(bpc_list):
        blob_to_class += [cls] * bpc
    blob_to_class = np.array(blob_to_class)
    y = blob_to_class[blob_ids]

    # --- 3) Nonlinear warp to break linear separability ---
    def nonlinear_warp(X, strength):
        Z = X.copy()
        if n_features < 2:
            # if only 1D, embed a fake second axis to twist then project back
            pad = np.zeros((Z.shape[0], 1))
            Z = np.hstack([Z, pad])

        # use first two dims as a canvas for warping
        x0 = Z[:, 0]
        x1 = Z[:, 1]

        # sinusoidal shear + circular swirl
        # strength scales with (1/separability) so harder sets get *more* warp
        s = strength * (1.5 / (separability + 0.2))
        x1_new = x1 + s * np.sin(x0)
        r2 = x0**2 + x1_new**2
        theta = s * 0.2 * np.sin(r2**0.5)

        # rotate by theta (point-wise)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0_rot = cos_t * x0 - sin_t * x1_new
        x1_rot = sin_t * x0 + cos_t * x1_new

        Z[:, 0] = x0_rot
        Z[:, 1] = x1_rot
        return Z[:, :n_features]  # keep original dimensionality

    X = nonlinear_warp(X, strength=1.0)

    # mix a tiny bit of label noise at low separability to further foil hyperplanes
    if separability < 0.8:
        flip_mask = rng.rand(len(y)) < (0.02 + 0.05 * (0.8 - separability))
        y[flip_mask] = rng.randint(0, n_classes, size=flip_mask.sum())

    # standardize features (nice for models/plots)
    X = StandardScaler().fit_transform(X)

    # --- 4) (Optional) verify nonlinearity, bump warp if needed ---
    if ensure_nonlinear:
        for attempt in range(max_retries):
            acc = LogisticRegression(max_iter=200, multi_class='auto').fit(X, y).score(X, y)
            if acc < 0.98:  # good enough: linear can't perfectly separate
                break
            # strengthen warp and try again
            X = nonlinear_warp(X, strength=1.0 + 0.8 * (attempt + 1))
            X = StandardScaler().fit_transform(X)

    return X, y