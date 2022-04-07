""" Trainer and evaluator of the networks. """
from collections import OrderedDict
from pathlib import PosixPath

import torch
import torch.nn.functional as F
from performer_pytorch.performer_pytorch import find_modules, FastAttention
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.fft import fftn
from tqdm import tqdm

from util import log_reconstructions


def fft_abs(x):
    img_torch = (x + 1.) / 2.
    fft = fftn(img_torch, norm="ortho")
    return torch.abs(fft)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ----------------------------------------------------------------------------------------------------------------------
# VQVAE
# ----------------------------------------------------------------------------------------------------------------------
def train_vqvae(
        model,
        start_epoch: int,
        best_loss: float,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        n_epochs: int,
        eval_freq: int,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: PosixPath,
):
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_vqvae(
        model=model,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_vqvae(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
        )
        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_vqvae(
                model=model,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / "best_model.pth"))

        if (epoch + 1) == 20:
            print(f"Changing codebook decay to 0.7")
            raw_model.codebook.decay = 0.70
        if (epoch + 1) == 40:
            print(f"Changing codebook decay to 0.8")
            raw_model.codebook.decay = 0.80
        if (epoch + 1) == 80:
            print(f"Changing codebook decay to 0.95")
            raw_model.codebook.decay = 0.95
        if (epoch + 1) == 100:
            print(f"Changing codebook decay to 0.99")
            raw_model.codebook.decay = 0.99

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))

    return val_loss


def train_epoch_vqvae(
        model,
        loader,
        optimizer,
        scheduler,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        scaler: GradScaler,
):
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader))

    for step, x in pbar:
        img = x["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            x_tilde, latent_loss, perplexity_code, _ = model(img)

            loss_recon = F.smooth_l1_loss(x_tilde.float(), img.float())

            loss = loss_recon + latent_loss

            loss = loss.mean()
            loss_recon = loss_recon.mean()
            latent_loss = latent_loss.mean()
            perplexity_code = perplexity_code.mean()

            losses = OrderedDict(
                loss=loss,
                loss_recon=loss_recon,
                loss_latent=latent_loss,
                perplexity=perplexity_code
            )

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "loss_recon": f"{losses['loss_recon'].item():.6f}",
                "perplexity": f"{losses['perplexity'].item():.6f}",
                "lr": f"{get_lr(optimizer):.6f}"
            },
        )


@torch.no_grad()
def eval_vqvae(
        model,
        loader,
        device,
        step: int,
        writer: SummaryWriter,
):
    raw_model = model.module if hasattr(model, "module") else model

    model.eval()

    total_losses = OrderedDict()

    for x in loader:
        img = x["image"].to(device)

        with autocast(enabled=True):
            x_tilde, latent_loss, perplexity_code, _ = model(img)

            loss_recon = F.smooth_l1_loss(x_tilde.float(), img.float())

            loss = loss_recon + latent_loss

            loss = loss.mean()
            loss_recon = loss_recon.mean()
            latent_loss = latent_loss.mean()
            perplexity_code = perplexity_code.mean()

            losses = OrderedDict(
                loss=loss,
                loss_recon=loss_recon,
                loss_latent=latent_loss,
                perplexity=perplexity_code
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        x=x,
        model=raw_model,
        device=device,
        writer=writer,
        step=step
    )

    return total_losses["loss_recon"]


# ----------------------------------------------------------------------------------------------------------------------
# VQGAN
# ----------------------------------------------------------------------------------------------------------------------
def train_vqgan(
        model,
        discriminator,
        start_epoch: int,
        best_loss: float,
        train_loader,
        val_loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        n_epochs: int,
        eval_freq: int,
        writer_train: SummaryWriter,
        writer_val: SummaryWriter,
        device: torch.device,
        run_dir: PosixPath,
        gan_criterion,
        squeeze_criterion,
        gan_weight: float,
        fft_weight: float,
        squeeze_weight: float,
):
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_vqgan(
        model=model,
        discriminator=discriminator,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val,
        gan_criterion=gan_criterion,
        squeeze_criterion=squeeze_criterion,
        gan_weight=gan_weight,
        fft_weight=fft_weight,
        squeeze_weight=squeeze_weight,
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")
    for epoch in range(start_epoch, n_epochs):
        train_epoch_vqgan(
            model=model,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_g=optimizer_g,
            scheduler_g=scheduler_g,
            optimizer_d=optimizer_d,
            scheduler_d=scheduler_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            gan_criterion=gan_criterion,
            squeeze_criterion=squeeze_criterion,
            gan_weight=gan_weight,
            fft_weight=fft_weight,
            squeeze_weight=squeeze_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_vqgan(
                model=model,
                discriminator=discriminator,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                gan_criterion=gan_criterion,
                squeeze_criterion=squeeze_criterion,
                gan_weight=gan_weight,
                fft_weight=fft_weight,
                squeeze_weight=squeeze_weight,
            )
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / 'best_model.pth'))

        if (epoch + 1) == 40:
            print(f"Changing codebook decay to 0.7")
            raw_model.codebook.decay = 0.70
        if (epoch + 1) == 80:
            print(f"Changing codebook decay to 0.8")
            raw_model.codebook.decay = 0.80
        if (epoch + 1) == 160:
            print(f"Changing codebook decay to 0.95")
            raw_model.codebook.decay = 0.95
        if (epoch + 1) == 200:
            print(f"Changing codebook decay to 0.99")
            raw_model.codebook.decay = 0.99

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / 'final_model.pth'))

    return val_loss


def train_epoch_vqgan(
        model,
        discriminator,
        loader,
        optimizer_g,
        scheduler_g,
        optimizer_d,
        scheduler_d,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        gan_criterion,
        squeeze_criterion,
        gan_weight: float,
        fft_weight: float,
        squeeze_weight: float,
        scaler_g,
        scaler_d,
):
    model.train()
    discriminator.train()

    pbar = tqdm(enumerate(loader), total=len(loader))

    for step, x in pbar:
        img = x["image"].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            x_tilde, latent_loss, perplexity_code, _ = model(img)

            l1_loss = F.l1_loss(x_tilde.float(), img.float())

            fft_loss = F.mse_loss(fft_abs(x_tilde.float()), fft_abs(img.float()))

            squeeze_loss = squeeze_criterion(x_tilde.float(), img.float())

            logits_fake = discriminator(x_tilde.contiguous().float())
            g_loss = gan_criterion(logits_fake, True)

            loss = l1_loss + latent_loss + fft_weight * fft_loss + squeeze_weight * squeeze_loss + gan_weight * g_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            fft_loss = fft_loss.mean()
            squeeze_loss = squeeze_loss.mean()
            latent_loss = latent_loss.mean()
            g_loss = g_loss.mean()
            perplexity_code = perplexity_code.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                fft_loss=fft_loss,
                squeeze_loss=squeeze_loss,
                loss_latent=latent_loss,
                g_loss=g_loss,
                perplexity=perplexity_code
            )

        scaler_g.scale(losses['loss']).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        scheduler_g.step()

        # DISCRIMINATOR
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            logits_fake = discriminator(x_tilde.contiguous().detach())
            loss_d_fake = gan_criterion(logits_fake, False)

            logits_real = discriminator(img.contiguous().detach())
            loss_d_real = gan_criterion(logits_real, True)

            d_loss = gan_weight * (loss_d_fake + loss_d_real) * 0.5

            d_loss = d_loss.mean()

        scaler_d.scale(d_loss).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
        scheduler_d.step()

        losses["d_loss"] = d_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "fft_loss": f"{losses['fft_loss'].item():.6f}",
                "squeeze_loss": f"{losses['squeeze_loss'].item():.6f}",
                "perplexity": f"{losses['perplexity'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            },
        )


@torch.no_grad()
def eval_vqgan(
        model,
        discriminator,
        loader,
        device,
        step,
        writer,
        gan_criterion,
        squeeze_criterion,
        gan_weight,
        fft_weight,
        squeeze_weight,
):
    raw_model = model.module if hasattr(model, "module") else model

    model.eval()
    discriminator.eval()

    total_losses = OrderedDict()

    for x in loader:
        img = x["image"].to(device)

        with autocast(enabled=True):
            x_tilde, latent_loss, perplexity_code, _ = model(img)

            l1_loss = F.l1_loss(x_tilde.float(), img.float())

            fft_loss = F.mse_loss(fft_abs(x_tilde.float()), fft_abs(img.float()))

            squeeze_loss = squeeze_criterion(x_tilde.float(), img.float())

            logits_fake = discriminator(x_tilde.contiguous().float())
            g_loss = gan_criterion(logits_fake, True)

            logits_fake = discriminator(x_tilde.contiguous().detach())
            loss_d_fake = gan_criterion(logits_fake, False)
            logits_real = discriminator(img.contiguous().detach())
            loss_d_real = gan_criterion(logits_real, True)
            d_loss = (loss_d_fake + loss_d_real) * 0.5

            loss = l1_loss + latent_loss + fft_weight * fft_loss + squeeze_weight * squeeze_loss + gan_weight * g_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            fft_loss = fft_loss.mean()
            squeeze_loss = squeeze_loss.mean()
            latent_loss = latent_loss.mean()
            g_loss = g_loss.mean()
            d_loss = d_loss.mean()
            perplexity_code = perplexity_code.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                fft_loss=fft_loss,
                squeeze_loss=squeeze_loss,
                loss_latent=latent_loss,
                g_loss=g_loss,
                d_loss=d_loss,
                perplexity=perplexity_code
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    log_reconstructions(
        x=x,
        model=raw_model,
        device=device,
        writer=writer,
        step=step
    )

    return total_losses['l1_loss']


# ----------------------------------------------------------------------------------------------------------------------
# Performer
# ----------------------------------------------------------------------------------------------------------------------
def train_performer(
        model,
        vqvae,
        start_epoch,
        best_loss,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        n_epochs,
        eval_freq,
        writer_train,
        writer_val,
        device,
        run_dir,
        redrawn_freq,
):
    raw_model = model.module if hasattr(model, "module") else model

    val_loss = eval_performer(
        model=model,
        vqvae=vqvae,
        loader=val_loader,
        device=device,
        step=len(train_loader) * start_epoch,
        writer=writer_val
    )
    print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_performer(
            model=model,
            vqvae=vqvae,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer_train
        )

        if (epoch + 1) % redrawn_freq == 0:
            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            raw_model.model.performer.proj_updater.calls_since_last_redraw.zero_()

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_performer(
                model=model,
                vqvae=vqvae,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    'best_loss': best_loss,
                }

                # save checkpoint
                torch.save(checkpoint, str(run_dir / "checkpoint_best.pth"))

                torch.save(raw_model.state_dict(), str(run_dir / 'best_model.pth'))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / 'final_model.pth'))

    return val_loss


def train_epoch_performer(
        model,
        vqvae,
        loader,
        optimizer,
        scheduler,
        device,
        epoch,
        writer
):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["image"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            encoded = raw_vqvae.encode_code(img)
            encoded = encoded.reshape(encoded.shape[0], -1)
            encoded = encoded[:, raw_model.ordering.index_sequence]
            encoded = F.pad(encoded, (1, 0), "constant", raw_vqvae.n_embed)

            encoded_in = encoded[:, :-1]
            encoded_out = encoded[:, 1:]

        logits = model(encoded_in)

        loss = F.cross_entropy(logits.transpose(1, 2), encoded_out)
        losses = OrderedDict(loss=loss)

        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}"
            }
        )


@torch.no_grad()
def eval_performer(
        model,
        vqvae,
        loader,
        device,
        step,
        writer
):
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    total_losses = OrderedDict()

    for x in loader:
        img = x["image"].to(device)
        encoded = raw_vqvae.encode_code(img)
        encoded = encoded.reshape(encoded.shape[0], -1)
        encoded = encoded[:, raw_model.ordering.index_sequence]
        encoded = F.pad(encoded, (1, 0), "constant", raw_vqvae.n_embed)

        encoded_in = encoded[:, :-1]
        encoded_out = encoded[:, 1:]

        logits = model(encoded_in)

        loss = F.cross_entropy(logits.transpose(1, 2), encoded_out)
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

    if step == 0:
        print(f"Ordering size")
        print(raw_model.ordering.index_sequence.shape[0])
        print(f"Encoded size")
        print(encoded.shape[1] - 1)
        assert raw_model.ordering.index_sequence.shape[0] == (encoded.shape[1] - 1)

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)

    return total_losses['loss']
