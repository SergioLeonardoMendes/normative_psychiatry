import argparse
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.optim as optim
from monai.config import print_config
from monai.utils import set_determinism
from tensorboardX import SummaryWriter

from models.vqvae import VQVAE
from training_functions import train_vqvae
from util import get_training_data_loader, log_mlflow

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    # model
    parser.add_argument("--n_embed", default=8, type=int, help="Size of discrete latent space (K-way categorical).")
    parser.add_argument("--embed_dim", default=64, type=int, help="Dimensionality of each latent embedding vector.")
    parser.add_argument("--n_alpha_channels", default=2, type=int, help="Number of alpha channels in input image.")
    parser.add_argument("--n_channels", default=128, type=int, help="Number of channels in the encoder and decoder.")
    parser.add_argument("--n_res_channels", default=128, type=int, help="Number of channels in the residual layers.")
    parser.add_argument("--n_res_layers", default=2, type=int, help="Number of residual blocks.")
    parser.add_argument("--p_dropout", default=0.1, type=float, help="Number of strided convolutions")
    parser.add_argument("--latent_resolution", default="low", type=str, help="Number of strided convolutions")
    parser.add_argument("--vq_decay", type=float, default=0.99, help="vq_decay.")
    parser.add_argument("--commitment_cost", default=0.25, type=float, help="Number of strided convolutions")
    # training param
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--lr_decay", type=float, default=0.99999, help="Learning rate decay.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to betweeen evaluations.")
    parser.add_argument("--vbm_img", type=int, default=1, help="Use vbm preprocessed image, 1 (True) or 0 (False).")
    parser.add_argument("--augmentation", type=int, default=1, help="Use of augmentation, 1 (True) or 0 (False).")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    if args.latent_resolution in ["low", "high"]:
        cache_dir = output_dir / "cached_data_low"
    elif args.latent_resolution == "mid":
        cache_dir = output_dir / "cached_data_mid"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_training_data_loader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        augmentation=bool(args.augmentation),
        latent_resolution=args.latent_resolution,
        num_workers=args.num_workers,
        vbm_img=args.vbm_img
    )

    print("Creating model...")
    model = VQVAE(
        n_embed=args.n_embed,
        embed_dim=args.embed_dim,
        n_alpha_channels=args.n_alpha_channels,
        n_channels=args.n_channels,
        n_res_channels=args.n_res_channels,
        n_res_layers=args.n_res_layers,
        p_dropout=args.p_dropout,
        vq_decay=args.vq_decay,
        commitment_cost=args.commitment_cost,
        latent_resolution=args.latent_resolution,
    )

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_vqvae(
        model=model,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
    )

    log_mlflow(
        model=model,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    # args = parse_args()
    args_cfg = OmegaConf.load("config/vqvae.yaml")
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_cfg, args_cli)
    main(args)
