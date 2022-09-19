### Revert ordering and save likelihoods

import sys
sys.path.append('/project/src/python/')

from pathlib import Path
import mlflow.pytorch
import torch
import numpy as np

run_dir="get_likelihood_ADHD200_8"
transformer_uri="/project/mlruns/0/281c9ece0d8f44d6b05f4ce1836fda35/artifacts/final_model"

output_dir = Path("/project/outputs/")
run_dir = output_dir / run_dir
sel_prob_path = run_dir / "selected_probs"
cache_dir = output_dir / "cached_data_low"

transformer = mlflow.pytorch.load_model(transformer_uri, map_location=torch.device('cpu'))
transformer.eval()

for i in range(922):
    selected_subject_npy = f"{sel_prob_path}/selected_probs_{i}.npy"
    subject_probs = np.load(selected_subject_npy, allow_pickle=True)
    subject_probs_ord = np.zeros(shape=subject_probs.shape)
    subject_probs_ord[0,:] = subject_probs[0,transformer.ordering.revert_ordering]
    np.save(sel_prob_path / f'ord_selected_probs_{i}.npy', subject_probs_ord)

