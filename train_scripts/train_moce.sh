master_addr="127.0.0.1"
node_rank=1
pretrained_path="meta-llama/Llama-2-7b-chat-hf"
data_path='../data/elbow_data/it_data_add_instructor.jsonl'
output_path='moce_llama2_7b_inst'

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup torchrun --master_port=4742 --nproc_per_node 4 ../train_moce.py \
    --model_name_or_path $pretrained_path \
    --data_path $data_path \
    --bf16 True \
    --num_clusters 4 \
    --model_max_length 512 \
    --output_dir $output_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate 2e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_steps 200 \
    --tf32 True > ${output_path}.log &
