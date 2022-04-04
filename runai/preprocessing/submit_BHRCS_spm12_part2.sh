runai submit \
  --name sgospm12-bhrcs-p2 \
  --image 10.202.67.207:5000/wds20:sgospm12 \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 1 \
  --large-shm \
  --node-type dgx2-a \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/project/data/INPD/rawtovbm:/data \
  --command -- /opt/spm12/spm12 batch /data/start_spm12_job2.m