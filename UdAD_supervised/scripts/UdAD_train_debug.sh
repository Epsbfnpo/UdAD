set -ex
python3 /home/sheng/used_by_zhenlin/UdAD/train.py \
--dataroot /home/sheng/Diffusion/data \
--checkpoints_dir /home/sheng/used_by_zhenlin/UdAD/checkpoints_debug \
--name hcpUdAD_debug_zhenlin \
--dataset_mode hcpUAD \
--num_threads 1 \
--batch_size 1 \
--input_patch_size 32 \
--data_norm instance_norm_3D \
--model UdAD \
--input_nc 7  \
--output_nc 1  \
--output_nc2 2  \
--cnum 8 \
--n_epochs 1 \
--n_epochs_decay 0 \
--save_epoch_freq 5 \
--gpu_ids 0