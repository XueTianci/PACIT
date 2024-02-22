N_GPU=8
gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=1234

PROJECT_NAME="Llama-2-7b-hf_seed_42_train_60_test_100"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
fsdp="LlamaDecoderLayer"
NUM_EXP=1


#################### PACIT-GT #####################
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WORLD_SIZE=$N_GPU CUDA_VISIBLE_DEVICES=$gpu_vis torchrun --nproc_per_node $N_GPU --master_port $MASTER_PORT \
    instruction_tuning.py \
    --model_name_or_path $MODEL_NAME \
    --train_data_path "./data/seed-42/defintion_pos_1_neg_1_train_60_test_100/held_out/train/PACIT-few-shot-GT.jsonl" \
    --output_dir "./output" \
    --cache_dir "./" \
    --num_train_epochs 5 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --bf16 True \
    --learning_rate 2e-5 \
    --save_strategy "epoch" \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --quantify False \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}" \
    --set_seed 42 \
    --project_name $PROJECT_NAME \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap $fsdp \

#################### PACIT-Random #####################
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WORLD_SIZE=$N_GPU CUDA_VISIBLE_DEVICES=$gpu_vis torchrun --nproc_per_node $N_GPU --master_port $MASTER_PORT \
    instruction_tuning.py \
    --model_name_or_path $MODEL_NAME \
    --train_data_path "./data/seed-42/defintion_pos_1_neg_1_train_60_test_100/held_out/train/PACIT-few-shot-Random.jsonl" \
    --output_dir "./output" \
    --cache_dir "./" \
    --num_train_epochs 5 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --bf16 True \
    --learning_rate 2e-5 \
    --save_strategy "epoch" \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --quantify False \
    --run_name "PACIT_train_few-shot_Random_pos_${NUM_EXP}_neg_${NUM_EXP}" \
    --set_seed 42 \
    --project_name $PROJECT_NAME \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap $fsdp \

