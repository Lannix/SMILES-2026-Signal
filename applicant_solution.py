import os
import json
import gdown
import random

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

set_seed(42)


# Download the dataset
if not os.path.exists('challenge.mat'):
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    downloaded_file = "challenge.mat"
    gdown.download(url, downloaded_file, quiet=False, fuzzy=True) # TypeError: download() got an unexpected keyword argument 'fuzzy'

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def rank1_from_matrix(matrix):
    """
    Find 1-rank component in SVD.
    """
    cov = matrix.conj().T @ matrix / matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = matrix @ vecs[:, -1] # project matrix onto eigenvector corresponding to the maximum eigenvalue
    denom = np.vdot(shared, shared) + 1e-30
    # restore rank-1 component for all channels
    return np.column_stack([
        (np.vdot(shared, matrix[:, ch]) / denom) * shared
        for ch in range(matrix.shape[1])
    ])

def your_canceller(tx_n, rx):
    # 1. find tx nonlinearity (without E correction) from baseline
    tx_pred = helpers["fit_tx_prediction"](rx)
    rx_res_with_E = rx - tx_pred
    
    # 2. 1-rank source E correction with SVD
    E_pred = rank1_from_matrix(rx_res_with_E)
    
    rx_hat = rx_res_with_E - E_pred
    
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
