docker run \
 -d \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1133:1134 \
 --name scf_train \
 --rm \
 --init \
 -v /home/l.erlygin/face-evaluation:/app \
 --gpus all \
 -w="/app" \
 face-eval \
 python evaluation/ijb_evals.py
