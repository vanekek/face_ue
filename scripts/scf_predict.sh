docker run \
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
 python trainers/train_scf.py predict \
 --config configs/train/train_sphere_face.yaml \
 --ckpt_path=/app/models/scf/epoch=3-step=90000.ckpt \
 --trainer.devices=1