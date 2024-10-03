set -ex
python3 /home/sheng/used_by_zhenlin/UdAD_supervised/train.py \
--dataroot /home/sheng/used_by_zhenlin \
--checkpoints_dir /home/sheng/used_by_zhenlin/UdAD_supervised/checkpoints \
--name hcpUdAD_100_zhenlin_supervised \
--dataset_mode hcpUAD \
--num_threads 1 \
--batch_size 1 \
--input_patch_size -1 \
--data_norm instance_norm_3D \
--model UdAD \
--input_nc 7  \
--output_nc 3  \
--output_nc2 3  \
--cnum 8 \
--n_epochs 100 \
--n_epochs_decay 0 \
--save_epoch_freq 5 \
--gpu_ids 0