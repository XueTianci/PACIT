data_dir=../data/splits/default
task_dir=../data/tasks/
output_dir=../../data/seed-42


python ../src/convert_data_to_s2s.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 60 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 1 \
    --num_neg_examples 1 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_pos_1_neg_1_train_60_test_100/ \
    --model_name google/t5-large-lm-adapt \
    --set_seed 42 \