from omegaconf import OmegaConf
from pathlib import Path

import mlflow.pytorch
import torch
import torch.nn.functional as F
from monai.utils import set_determinism
import numpy as np

from util import get_training_data_loader

def main(args):
    set_determinism(seed=args.seed)

    output_dir = Path("/project/outputs/")
    output_dir.mkdir(exist_ok=True)

    run_dir = output_dir / args.run_dir
    run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    sel_prob_path = run_dir / "selected_probs"
    sel_prob_path.mkdir(exist_ok=True)
    print("Getting data...")
    if args.latent_resolution in ["low", "high"]:
        cache_dir = output_dir / "cached_data_low"
    elif args.latent_resolution == "mid":
        cache_dir = output_dir / "cached_data_mid"
    cache_dir.mkdir(exist_ok=True)

    val_loader = get_training_data_loader(
        cache_dir,
        batch_size=1,
        training_ids=None,
        validation_ids=args.val_list,
        only_val=True,
        augmentation=False,
        latent_resolution=args.latent_resolution,
        num_workers=4,
        vbm_img=args.vbm_img
    )

    device = torch.device("cuda")
    # Load VQVAE to produce the encoded samples
    print(f"Loading VQ-VAE from {args.vqvae_uri}")
    vqvae = mlflow.pytorch.load_model(args.vqvae_uri)
    vqvae.eval()
    vqvae.to(device)

    print(f"Loading transformer from {args.transformer_uri}")
    transformer = mlflow.pytorch.load_model(args.transformer_uri)
    transformer.eval()
    transformer.to(device)

    likelihood_ls = []
    with torch.no_grad():
        for idx, x in enumerate(val_loader):
            img = x["image"].to(device)
            encoded = vqvae.encode_code(img)
            encoded = encoded.reshape(encoded.shape[0], -1)
            encoded = encoded[:, transformer.ordering.index_sequence]
            encoded = F.pad(encoded, (1, 0), "constant", vqvae.n_embed)

            encoded_in = encoded[:, :-1]
            encoded_out = encoded[:, 1:]

            logits = transformer(encoded_in)
            probs = F.softmax(logits, dim=-1).cpu()
            selected_probs = torch.gather(probs, 2, encoded_out.cpu().unsqueeze(2).long())
            selected_probs = selected_probs.squeeze(2)
            selected_probs.cpu().numpy()
            np.save(sel_prob_path / f'selected_probs_{idx}.npy', selected_probs)
            likelihood = torch.sum(torch.log(selected_probs), dim=-1)
            likelihood.cpu().numpy()
            likelihood_ls.append(likelihood)

    np.save(run_dir / 'likelihood.npy', likelihood_ls)


if __name__ == "__main__":
    args_cfg = OmegaConf.load("config/transformer.yaml")
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_cfg, args_cli)

    main(args)