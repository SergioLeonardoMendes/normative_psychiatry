runai submit \
  --name sgo-normativepsy-bash \
  --image 10.202.67.207:5000/wds20:normativepsychiatry \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 4 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/normative_psychiatry:/project \
  --volume /nfs/home/wds20/sergio/project/data/:/data \
  --command -- sleep infinity