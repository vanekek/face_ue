docker run \
 --shm-size=8g \
 --memory=80g \
 --cpus=40 \
 --user 1133:1134 \
 --name scf_predict \
 --rm \
 --init \
 -v /home/l.erlygin/face_ue:/app \
 --gpus all \
 -w="/app" \
 face-eval \
 python trainers/train.py predict \
 --config configs/train/train_sphere_face.yaml \
 --ckpt_path=/app/model_weights/scf/epoch=3-step=90000.ckpt \
 --trainer.devices=1