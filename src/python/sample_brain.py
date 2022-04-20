from omegaconf import OmegaConf
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.pytorch
import mlflow
import nibabel as nib
import numpy as np
import torch
from monai.config import print_config
from monai.utils import set_determinism


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("/project/outputs/")
    output_dir.mkdir(exist_ok=True)
    run_dir = output_dir / args.run_dir
    run_dir.mkdir(exist_ok=True)

    # Load Transformer
    device = torch.device("cuda")
    transformer = mlflow.pytorch.load_model(args.transformer_uri)
    transformer.eval()
    transformer = transformer.to(device)

    # Load VQVAE
    vqvae = mlflow.pytorch.load_model(args.vqvae_uri)
    vqvae.eval()
    vqvae = vqvae.to(device)
    print(">>>Models loaded")

    # Sample Transformer
    n_samples = 1
    start_pixel = np.array([[vqvae.n_embed]])
    start_pixel = np.repeat(start_pixel, n_samples, axis=0)
    initial = torch.from_numpy(start_pixel).to(device)

    print(">>>Sampling...")
    latent_code = transformer.sample(prefix=initial, sample=True)

    # Saving latent code
    np.save(f"{str(run_dir)}/transformer_latent_code.npy", latent_code.cpu().numpy())
    print(f"Saved latent representation at {str(run_dir)}/transformer_latent_code.npy")

    # Decoding latent code
    with torch.no_grad():
        recons = vqvae.decode_code(latent_code.to(device).long()).cpu().numpy()

    # Preparing image data
    data = []
    if not args.vbm_img:
        norm_data = np.clip((recons[0, 0] - recons[0, 0].min()) / (recons[0, 0].max() - recons[0, 0].min()), 0, 1) * 255
        data.append(norm_data)
        prefix = ['img']
        affine = np.eye(4)
    else:
        data.append(recons[0, 0])
        data.append(recons[0, 1])
        prefix = ['gm', 'wm']
        affine = np.diag([-1, 1, 1, 1])

    # Saving files
    for i, file in enumerate(prefix):
        nii = nib.Nifti1Image(data[i], affine=affine)
        nib.save(nii, f'{str(run_dir)}/{prefix}_sample_transformer.nii.gz')
        print(f"Sample created at {str(run_dir)}/{prefix}_sample_transformer.nii.gz")
        plt.imshow(data[i][:, :, 100])
        plt.savefig(f'{str(run_dir)}/{prefix}_sample_transformer_slice100.png')
        print(f"image created at {str(run_dir)}/{prefix}_sample_transformer_slice100.png")


if __name__ == "__main__":
    args = OmegaConf.from_cli()
    main(args)
