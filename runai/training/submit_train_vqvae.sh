#!/bin/bash
seed=2
run_dir="vqvae_low_abcd"
training_ids="/project/data/train_participants_ABCD.tsv"
validation_ids="/project/data/valid_participants_ABCD.tsv"
n_embed=64
embed_dim=64
n_alpha_channels=2
n_channels=256
n_res_channels=256
n_res_layers=3
p_dropout=0.1
latent_resolution="low"
vq_decay=0.5
commitment_cost=0.25
batch_size=12
lr=0.0005
lr_decay=0.9999
n_epochs=50
eval_freq=2
augmentation=0
vbm_img=1
num_workers=16
experiment="VQVAE_low_ABCD"

runai submit \
  --name sgo-vqvae-low-abcd-3 \
  --image 10.202.67.207:5000/wds20:normativepsychiatry \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 8 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/normative_psychiatry:/project \
  --volume /nfs/home/wds20/sergio/project/data/:/data \
  --command -- python3 /project/src/python/train_vqvae.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      n_embed=${n_embed} \
      embed_dim=${embed_dim} \
      n_alpha_channels=${n_alpha_channels} \
      n_channels=${n_channels} \
      n_res_channels=${n_res_channels} \
      n_res_layers=${n_res_layers} \
      p_dropout=${p_dropout} \
      latent_resolution=${latent_resolution} \
      commitment_cost=${commitment_cost} \
      vq_decay=${vq_decay} \
      batch_size=${batch_size} \
      lr=${lr} \
      lr_decay=${lr_decay} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      vbm_img=${vbm_img} \
      augmentation=${augmentation} \
      num_workers=${num_workers} \
      experiment=${experiment}
