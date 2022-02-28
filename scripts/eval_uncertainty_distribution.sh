# Scale with batching
python3 ./face_lib/evaluation/dataset_distribution.py \
  --checkpoint_path=/gpfs/data/gpfs0/k.fedyanin/space/models/scale/02_sigm_mul_selection/64/checkpoint.pth \
  --dataset_path=/gpfs/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/big \
  --image_paths_table=/gpfs/data/gpfs0/k.fedyanin/space/IJB/aligned_data_for_fusion/images_lists/big.txt \
  --dataset_name=IJBC \
  --config_path=./configs/scale/02_sigm_mul_coef_selection/64.yaml \
  --batch_size=64 \
  --uncertainty_strategy=scale \
  --device_id=0 \
  --save_fig_path=/beegfs/home/r.kail/faces/figures/22_dataset_distribution/test