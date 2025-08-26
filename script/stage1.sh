export HF_HOME="/tmp/.cache/huggingface"
# modify as you will
export LLAMA_MODEL_NAME="meta-llama/Llama-3.1-8B"
export CKPT_DIR="Llama-3.1-8b_stage1_lora_rank32_bsz_4_epoch3_lr1e-4_dropout0"
export HF_TOKEN="Your Huggingface Token"

# If you have previously downloaded the LLM, you can choose to not skip this command and point HF_HOME to your download path
mkdir -p /tmp/.cache/huggingface
huggingface-cli download $LLAMA_MODEL_NAME --exclude "original/*"

llamafactory-cli train \
  --hf_hub_token "$HF_TOKEN" \
  --model_name_or_path "$LLAMA_MODEL_NAME" \
  --trust_remote_code true \
  --enable_liger_kernel true \
  --use_unsloth_gc true \
  --torch_empty_cache_steps 1 \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --lora_rank 32 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj" \
  --deepspeed "examples/deepspeed/ds_z3_config.json" \
  --dataset_dir "checkpoint_llamafactory" \
  --dataset "stage1_train" \
  --eval_dataset "stage1_eval" \
  --template "llama3" \
  --cutoff_len 2048 \
  --overwrite_cache true \
  --output_dir "/tmp/$CKPT_DIR/" \
  --logging_steps 1 \
  --save_strategy "no" \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "1.0e-4" \
  --num_train_epochs "3.0" \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --per_device_eval_batch_size 4 \
  --eval_strategy "steps" \
  --eval_steps 10


llamafactory-cli "export" \
  --hf_hub_token "$HF_TOKEN" \
  --model_name_or_path "$LLAMA_MODEL_NAME" \
  --trust_remote_code true \
  --adapter_name_or_path "/tmp/$CKPT_DIR/" \
  --template "llama3" \
  --export_dir "/tmp/$CKPT_DIR/lora_merge" \
  --export_size 5 \
  --export_device "cpu"


python scripts/vllm_infer.py \
  --model_name_or_path "/tmp/$CKPT_DIR/lora_merge" \
  --dataset_dir checkpoint_llamafactory \
  --dataset stage1_eval \
  --template llama3 \
  --temperature 0 \
  --save_name "/tmp/$CKPT_DIR/pred.jsonl"

rsync -av --progress "/tmp/$CKPT_DIR/" "checkpoint_llamafactory/$CKPT_DIR" --exclude "lora_merge"