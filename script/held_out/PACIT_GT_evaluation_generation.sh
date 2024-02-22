gpu_vis=0
HELD_IN_OR_OUT="held_out"

PROJECT_NAME="Llama-2-7b-hf_seed_42_train_60_test_100"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
CHECKPOINT="checkpoint-420"
NUM_EXP=1
##################################### evaluation for generation ######################################

#evaluation for zero-shot
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_zero-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/SuperNI-zero-shot.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"

CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_zero-shot_Random" \
    --model_name_or_path "./output/$PROJECT_NAME/PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/PACIT-zero-shot-inference-Random.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --decoder_input True \
    --eval_method "generation"

#evaluation for few-shot

#self-generation
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_few-shot" \
    --model_name_or_path "./output/$PROJECT_NAME/PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/PACIT-few-shot-GT.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --eval_method "generation"

#with ground truth label
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_few-shot_GT" \
    --model_name_or_path "./output/$PROJECT_NAME/PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/PACIT-few-shot-inference-GT.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --decoder_input True \
    --eval_method "generation"

#with random label
CUDA_VISIBLE_DEVICES=$gpu_vis python \
    evaluation.py \
    --random_seed 42 \
    --output_dir "./output" \
    --project_name $PROJECT_NAME \
    --run_name "PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${HELD_IN_OR_OUT}/inference_few-shot_Random" \
    --model_name_or_path "./output/$PROJECT_NAME/PACIT_train_few-shot_GT_pos_${NUM_EXP}_neg_${NUM_EXP}/${CHECKPOINT}/pytorch_model.bin" \
    --data_path "./data/seed-42/defintion_pos_${NUM_EXP}_neg_${NUM_EXP}_train_60_test_100/${HELD_IN_OR_OUT}/test/PACIT-few-shot-inference-Random.jsonl" \
    --eval_batch_size 1 \
    --max_new_tokens 128 \
    --temperature 0 \
    --max_length 1024 \
    --tokenizer $MODEL_PATH \
    --decoder_input True \
    --eval_method "generation"