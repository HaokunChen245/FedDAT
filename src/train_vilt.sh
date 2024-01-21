CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false accelerate launch \
--config_file accelerate_config.yaml \
src/train/main.py \
--encoder_name vilt \
--pretrained_model_name ./models/vilt-b32-mlm \
--climb_data_dir ''  \
--do_train  \
--model_path ./models \
--output_dir ./logs  \
--batch_size 2 \
--val_batch_size 2 \
--comm_round 30 \
--local_epochs 1 \
--lr 1e-4  \
--optimizer_mode dat \
--seed 1 \
--adapter_reduction_factor 16 \
--adapter_config pfeiffer \
--splits train_small val test_small \
--ordered_cl_tasks domain