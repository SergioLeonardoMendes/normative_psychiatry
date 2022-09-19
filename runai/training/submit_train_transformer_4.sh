#!/bin/bash
seed=2
run_dir="transformerb_00010000_abcd"
training_ids="/project/data/train_participants_ABCD.tsv"
validation_ids="/project/data/valid_participants_ABCD.tsv"
input_height=24
input_width=28
input_depth=24
order_type="raster_scan"
transposed_1=0
transposed_2=0
transposed_3=0
transposed_4=1
transposed_5=0
reflected_rows=0
reflected_cols=0
reflected_depths=0
n_embd=64
n_layers=16
n_heads=8
local_attn_heads=4
local_window_size=256
ff_mult=4
ff_glu=0
rotary_position_emb=0
axial_position_emb=0
emb_dropout=0.3
ff_dropout=0.3
attn_dropout=0.3
latent_resolution="low"
redrawn_freq=3
vqvae_uri="/project/mlruns/0/f89cb9ee1a824324b04df9f4b7a8e430/artifacts/final_model"
batch_size=2
lr=0.001
lr_decay=0.999997
n_epochs=50
eval_freq=3
augmentation=0
vbm_img=1
num_workers=16
experiment="TRANSFORMERb_low_ABCD_00010000"

runai submit \
  --name sgo-transfb-abcd-00010000 \
  --image 10.202.67.207:5000/wds20:normativepsychiatry \
  --backoff-limit 0 \
  --gpu 2 \
  --cpu 8 \
  --node-type dgx2-a \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/normative_psychiatry:/project \
  --volume /nfs/home/wds20/sergio/project/data/:/data \
  --command -- python3 /project/src/python/train_transformer.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      input_height=${input_height} \
      input_width=${input_width} \
      input_depth=${input_depth} \
      order_type=${order_type} \
      transposed_1=${transposed_1} \
      transposed_2=${transposed_2} \
      transposed_3=${transposed_3} \
      transposed_4=${transposed_4} \
      transposed_5=${transposed_5} \
      reflected_rows=${reflected_rows} \
      reflected_cols=${reflected_cols} \
      reflected_depths=${reflected_depths} \
      n_embd=${n_embd} \
      n_layers=${n_layers} \
      n_heads=${n_heads} \
      local_attn_heads=${local_attn_heads} \
      local_window_size=${local_window_size} \
      ff_mult=${ff_mult} \
      ff_glu=${ff_glu} \
      rotary_position_emb=${rotary_position_emb} \
      axial_position_emb=${axial_position_emb} \
      emb_dropout=${emb_dropout} \
      ff_dropout=${ff_dropout} \
      attn_dropout=${attn_dropout} \
      latent_resolution=${latent_resolution} \
      redrawn_freq=${redrawn_freq} \
      vqvae_uri=${vqvae_uri} \
      batch_size=${batch_size} \
      lr=${lr} \
      lr_decay=${lr_decay} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      vbm_img=${vbm_img} \
      augmentation=${augmentation} \
      num_workers=${num_workers} \
      experiment=${experiment}
