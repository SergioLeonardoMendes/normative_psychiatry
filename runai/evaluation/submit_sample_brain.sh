#!/bin/bash
seed=2
run_dir="sample_brain_93d86cda9d064414885645a907ab4a83_aaee864e3e83473eb56953d8645ac991"
vqvae_uri="/project/mlruns/0/93d86cda9d064414885645a907ab4a83/artifacts/final_model"
transformer_uri="/project/mlruns/0/aaee864e3e83473eb56953d8645ac991/artifacts/final_model"
vbm_img=1

runai submit \
  --name sample-brain \
  --image 10.202.67.207:5000/wds20:normativepsychiatry \
  --backoff-limit 0 \
  --gpu 0.25 \
  --cpu 1 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/normative_psychiatry:/project \
  --command -- python /project/src/python/sample_brain.py \
  seed=${seed} \
  run_dir=${run_dir} \
  vqvae_uri=${vqvae_uri} \
  transformer_uri=${transformer_uri} \
  vbm_img=${vbm_img}