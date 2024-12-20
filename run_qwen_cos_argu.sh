#!/bin/bash
#!/bin/bash


name='eedi'
PATH_PRE="./"

# DATA_NAME="long_trn_bge_top100_recall_pairs.jsonl"
DATA_NAME="long_trn_sfr_r1_top100_recall_pairs.jsonl"
DOC_NAME="misconception_mapping.csv"
DATA_DIR=${PATH_PRE}/data/
MODEL_USE="recall33"
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/model_save/${MODEL_USE}_qlora_rerun
NUM_NEG_SAMPLES=8


#模型地址
# MODEL_PATH=./model_save/SFR-Embedding-2_R
MODEL_PATH=./model_save/qwen2.5-14b
# LORA_PATH=./model_save/recall1_qlora_rerun/epoch_19_model/adapter.bin
LORA_PATH=none
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3 simcse_deepspeed_qwen25_qlora_argu.py \
       --project_name ${name}_${MODEL_USE} \
       --train_data ${DATA_DIR}${DATA_NAME} \
       --doc_data ${DATA_DIR}${DOC_NAME} \
       --model_name_or_path ${MODEL_PATH} \
       --lora_path ${LORA_PATH} \
       --per_device_train_batch_size 3 \
       --per_device_eval_batch_size 3 \
       --train_group_size 4 \
       --num_neg_samples ${NUM_NEG_SAMPLES} \
       --gradient_accumulation_steps 1 \
       --query_max_len 256 \
       --passage_max_len 64 \
       --earystop 0 \
       --save_batch_steps 100000000000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 20  \
       --learning_rate 1e-4 \
       --num_warmup_steps 100 \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing
