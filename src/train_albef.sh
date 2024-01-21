CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false accelerate launch \
--config_file ./accelerate_config.yaml \
src/train/main.py \
--encoder_name albef_no_distill \
--pretrained_model_name ./models/ALBEF.pth \
--climb_data_dir ''  \
--do_train  \
--model_path ./models/ \
--output_dir ./logs/  \
--batch_size 2 \
--val_batch_size 2 \
--lr 1e-4  \
--optimizer_mode dat \
--seed 2 \
--adapter_reduction_factor 16 \
--adapter_config pfeiffer \
--splits train_small val test \
--ordered_cl_tasks domain







