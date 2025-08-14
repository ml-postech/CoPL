export WANDB_MODE=online
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export WANDB_PROJECT=CoPL_RM


# Set model_name to be 'google/gemma-2b-it' or 'google/gemma-7b-it' here
data_path='dataset/UF-P-2-10000-ALL.pkl'
model_name='google/gemma-2b-it'
embeds_path='gcf_user_embeds/UF-P-2-ALL-user_emb.pt'
log_dir='logs/CoPL_RM'
port_num=4788

# CUDA_LAUNCH_BLOCKING=1  #torchrun --nproc_per_node=4 --master_port 47700

torchrun --nproc_per_node=4 --master_port ${port_num} train_CoPL_rm.py \
        --model_name=${model_name} \
        --data_path=${data_path} \
        --user_embeds_path=${embeds_path} \
        --log_dir=${log_dir} \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --latent_dim 512 \
        --hidden_dim 512 \
        --learning_rate 5e-5 \
        --seed 0 \
        --gradient_checkpointing True \
        --wandbon True \
        --lora_r 8 \
        --num_experts 8 \
        --max_steps 1500 \
        --deepspeed scripts/ds_config.json 
