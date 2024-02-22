gpu_vis=0
HELD_IN_OR_OUT="held_out"

PROJECT_NAME="Llama-2-7b-hf_seed_42_train_60_test_100"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
CHECKPOINT="checkpoint-339"
NUM_EXP=1
##################################### evaluation for generation ######################################

############## training with zero-shot model #############
#evaluation for zero-shot
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "SuperNI_train_zero-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_zero-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/SuperNI_train_zero-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/SuperNI-zero-shot.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"

#evaluation for few-shot
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "SuperNI_train_zero-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_few-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/SuperNI_train_zero-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/SuperNI-few-shot.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"


############## training with few-shot model #############
#evaluation for zero-shot
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "SuperNI_train_few-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_zero-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/SuperNI_train_few-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/SuperNI-zero-shot.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"

#evaluation for few-shot
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "SuperNI_train_few-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_few-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/SuperNI_train_few-shot_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/SuperNI-few-shot.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"
    