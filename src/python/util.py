from pathlib import PosixPath

import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import start_run
from monai import transforms
from monai.data import PersistentDataset
from torch.utils.data import DataLoader


def log_mlflow(
        model,
        args,
        experiment: str,
        run_dir: PosixPath,
        val_loss: float,
):
    """Log model and performance on Mlflow system"""
    mlflow.set_tracking_uri("file:/project/mlruns")
    
    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        print(f"Setting mlflow experiment: {experiment}")
        mlflow.set_experiment(experiment)

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model

        mlflow.pytorch.log_model(raw_model, "final_model")
        raw_model.load_state_dict(torch.load(str(run_dir / 'best_model.pth')))
        mlflow.pytorch.log_model(raw_model, "best_model")


def get_data_dicts(
        ids_path: str,
        only_img: bool = False,
        shuffle: bool = False,
        vbm_img: bool = True
):
    """ Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")
    if shuffle:
        df = df.sample(frac=1, random_state=1)

    data_dicts = []
    for index, row in df.iterrows():
        if vbm_img:
            img_dict = {
                "gm": (
                    f"/data/{row['dataset']}/rawtovbm/images/{row['participant_id']}/{row['session_id']}/"
                    f"anat/smwc1{row['participant_id']}_{row['session_id']}_{row['run_id']}_T1w.nii"),
                "wm": (
                    f"/data/{row['dataset']}/rawtovbm/images/{row['participant_id']}/{row['session_id']}/"
                    f"anat/smwc2{row['participant_id']}_{row['session_id']}_{row['run_id']}_T1w.nii"),
                "csf": (
                    f"/data/{row['dataset']}/rawtovbm/images/{row['participant_id']}/{row['session_id']}/"
                    f"anat/smwc3{row['participant_id']}_{row['session_id']}_{row['run_id']}_T1w.nii")
            }
            info_dict = {
                "age": row['age'],
                "sex": row['sex'],
                "dataset": row['dataset']
            }
        else:
            pass

        result_dict = img_dict

        if not only_img:
            result_dict.update(info_dict)

        data_dicts.append(result_dict)

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_training_data_loader(
        cache_dir: str,
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        latent_resolution: str = "low",
        num_workers: int = 8,
        vbm_img: bool = True
):
    if latent_resolution in ["low", "high"]:
        resolution_size = [192, 224, 192]
    elif latent_resolution == "mid":
        resolution_size = [192, 228, 192]

    # Define transformations
    if vbm_img:
        result_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["gm", "wm"]),
                transforms.AddChanneld(keys=["gm", "wm"]),
                transforms.SpatialPadd(
                    keys=["gm", "wm"],
                    spatial_size=resolution_size,
                    method="symmetric",
                    mode="minimum"
                ),
                transforms.ConcatItemsd(keys=["gm", "wm"], name="image"),
                transforms.ToTensord(keys=['image'])
            ]
        )
    else:
        pass

    if augmentation:
        pass
    else:
        train_transforms = result_transforms
        val_transforms = result_transforms

    val_dicts = get_data_dicts(
        validation_ids,
        shuffle=False,
        vbm_img=vbm_img
    )
    val_ds = PersistentDataset(
        data=val_dicts,
        transform=val_transforms,
        cache_dir=cache_dir
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        training_ids,
        only_img=True,
        shuffle=False,
        vbm_img=vbm_img
    )
    train_ds = PersistentDataset(
        data=train_dicts,
        transform=train_transforms,
        cache_dir=cache_dir
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return train_loader, val_loader


def log_reconstructions(x, model, device, writer, step, use_patches=False):
    if use_patches:
        x_recon = model.reconstruct(x["image"].to(device))

        def log_figure(sample):
            img_row_1 = np.concatenate(
                (
                    x["image"][sample, 0, :, :, 12].cpu().numpy(),
                    x_recon[sample, 0, :, :, 12].cpu().numpy(),
                    x["image"][sample, 0, :, :, 24].cpu().numpy(),
                    x_recon[sample, 0, :, :, 24].cpu().numpy(),
                    x["image"][sample, 0, :, :, 36].cpu().numpy(),
                    x_recon[sample, 0, :, :, 36].cpu().numpy(),
                ),
                axis=1
            )

            img_row_2 = np.concatenate(
                (
                    x["image"][sample, 0, :, 12, :].cpu().numpy(),
                    x_recon[sample, 0, :, 12, :].cpu().numpy(),
                    x["image"][sample, 0, :, 24, :].cpu().numpy(),
                    x_recon[sample, 0, :, 24, :].cpu().numpy(),
                    x["image"][sample, 0, :, 36, :].cpu().numpy(),
                    x_recon[sample, 0, :, 36, :].cpu().numpy(),
                ),
                axis=1
            )

            img_row_3 = np.concatenate(
                (
                    x["image"][sample, 0, 12, :, :].cpu().numpy(),
                    x_recon[sample, 0, 12, :, :].cpu().numpy(),
                    x["image"][sample, 0, 24, :, :].cpu().numpy(),
                    x_recon[sample, 0, 24, :, :].cpu().numpy(),
                    x["image"][sample, 0, 36, :, :].cpu().numpy(),
                    x_recon[sample, 0, 36, :, :].cpu().numpy(),
                ),
                axis=1
            )

            img = np.concatenate(
                (
                    img_row_1,
                    img_row_2,
                    img_row_3,
                ),
                axis=0
            )

            fig = plt.figure(figsize=(5, 5), dpi=300)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            writer.add_figure(f'RECONSTRUCTION/{sample}', fig, step)

        log_figure(0)
    else:
        x_recon = model.reconstruct(x["image"].to(device))

        def log_figure(sample):
            fig, axs = plt.subplots(4, 2, figsize=(5, 5), dpi=300)

            axs[0, 0].set_title('Original')
            axs[0, 0].imshow(x["image"][sample, 0, :, :, 30].cpu(), cmap='gray')
            axs[0, 0].axis('off')
            axs[1, 0].imshow(x["image"][sample, 0, :, :, 90].cpu(), cmap='gray')
            axs[1, 0].axis('off')
            axs[2, 0].imshow(x["image"][sample, 0, 60, :, :].cpu(), cmap='gray')
            axs[2, 0].axis('off')
            axs[3, 0].imshow(x["image"][sample, 0, :, 60, :].cpu(), cmap='gray')
            axs[3, 0].axis('off')

            axs[0, 1].set_title('Reconstruction')
            axs[0, 1].imshow(x_recon.cpu()[sample, 0, :, :, 30], cmap='gray')
            axs[0, 1].axis('off')
            axs[1, 1].imshow(x_recon.cpu()[sample, 0, :, :, 90], cmap='gray')
            axs[1, 1].axis('off')
            axs[2, 1].imshow(x_recon.cpu()[sample, 0, 60, :, :], cmap='gray')
            axs[2, 1].axis('off')
            axs[3, 1].imshow(x_recon.cpu()[sample, 0, :, 60, :], cmap='gray')
            axs[3, 1].axis('off')
            writer.add_figure(f'RECONSTRUCTION/{sample}', fig, step)

        log_figure(0)
