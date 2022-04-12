""" Train minGPT on the VQ-VAEs priors. """
from omegaconf import OmegaConf
from pathlib import Path

import mlflow.pytorch
import torch
import torch.optim as optim
from monai.config import print_config
from monai.utils import set_determinism
from tensorboardX import SummaryWriter

from models.img2seq_ordering import Ordering3D
from models.performer import Performer
from training_functions import train_performer
from util import get_training_data_loader, log_mlflow


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True)

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
    if args.latent_resolution == "low":
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

    # Select ordering
    ordering = Ordering3D(
        dimensions=(args.input_height, args.input_width, args.input_depth),
        ordering_type="raster_scan",
        transposed_1=bool(args.transposed_1),
        transposed_2=bool(args.transposed_2),
        transposed_3=bool(args.transposed_3),
        transposed_4=bool(args.transposed_4),
        transposed_5=bool(args.transposed_5),
        reflected_rows=bool(args.reflected_rows),
        reflected_cols=bool(args.reflected_cols),
        reflected_depths=bool(args.reflected_depths),
    )

    device = torch.device("cuda")
    # Load VQVAE to produce the encoded samples
    print(f"Loading VQ-VAE from {args.vqvae_uri}")
    vqvae = mlflow.pytorch.load_model(args.vqvae_uri)
    vqvae.eval()

    # Create model
    print("Creating model...")
    model = Performer(
        ordering=ordering,
        vocab_size=vqvae.n_embed + 1,  # vqvae.n_embed == <BOS>
        max_seq_len=args.input_height * args.input_width * args.input_depth,
        n_embd=args.n_embd,
        n_layer=args.n_layers,
        n_heads=args.n_heads,
        local_attn_heads=args.local_attn_heads,
        local_window_size=args.local_window_size,
        ff_mult=args.ff_mult,
        ff_glu=bool(args.ff_glu),
        rotary_position_emb=bool(args.rotary_position_emb),
        axial_position_emb=bool(args.axial_position_emb),
        emb_dropout=args.emb_dropout,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
    )

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        vqvae = torch.nn.DataParallel(vqvae)

    model = model.to(device)
    vqvae = vqvae.to(device)

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
    val_loss = train_performer(
        model=model,
        vqvae=vqvae,
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
        redrawn_freq=args.redrawn_freq,
    )

    log_mlflow(
        model=model,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args_cfg = OmegaConf.load("config/transformer.yaml")
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_cfg, args_cli)
    main(args)