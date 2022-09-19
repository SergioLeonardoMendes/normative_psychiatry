#!/bin/bash
seed=2
run_dir="get_likelihood_ABIDE2_4"
val_list="/project/data/participants_ABIDE2.tsv"
latent_resolution="low"
vqvae_uri="/project/mlruns/0/f89cb9ee1a824324b04df9f4b7a8e430/artifacts/final_model"
transformer_uri="/project/mlruns/0/cb7a0db6883141d7b8c35bc8e9f7ecfd/artifacts/final_model"
clinical_dir="/project/data/participants_ABIDE2.tsv"
vbm_img=1

runai submit \
  --name sgo-get-likelihood-abide2-4 \
  --image 10.202.67.207:5000/wds20:normativepsychiatry \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/normative_psychiatry:/project \
  --volume /nfs/home/wds20/sergio/project/data/:/data \
  --command -- python3 /project/src/python/get_likelihood.py \
    seed=${seed} \
    run_dir=${run_dir} \
    val_list=${val_list} \
    latent_resolution=${latent_resolution} \
    vqvae_uri=${vqvae_uri} \
    transformer_uri=${transformer_uri} \
    clinical_dir=${clinical_dir} \
    vbm_img=${vbm_img} 
