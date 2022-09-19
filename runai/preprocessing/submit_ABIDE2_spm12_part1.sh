if [[ $# -ne 1 ]]; then
  echo "  Invalid number of arguments (1 expected)!"
  exit 2
fi

PARTN=$1
runai submit \
  --name sgospm12-abide2-p1-$PARTN \
  --image 10.202.67.207:5000/wds20:sgospm12 \
  --backoff-limit 0 \
  --gpu 0 \
  --cpu 1 \
  --large-shm \
  --node-type dgx1-3 \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/sergio/project/data/ABIDE2/rawtovbm:/data \
  --command -- /opt/spm12/spm12 batch /data/start_spm12_job1_x0$PARTN.m
