set -ex
python3 /home/sheng/used_by_zhenlin/UdAD/test.py \
--dataroot /home/sheng/Diffusion/data \
--checkpoints_dir /home/sheng/used_by_zhenlin/UdAD/checkpoints \
--results_dir /home/sheng/used_by_zhenlin/UdAD/results \
--eval \
--name hcpUdAD_100_zhenlin_disentangled_learning \
--epoch latest \
--dataset_mode hcpUAD \
--num_threads 0 \
--serial_batches \
--batch_size 1 \
--input_patch_size -1 \
--data_norm instance_norm_3D \
--model UdAD \
--input_nc 7 \
--output_nc 1 \
--cnum 8 \
--num_test 57 \
--save_prediction 1 \
--gpu_ids 1