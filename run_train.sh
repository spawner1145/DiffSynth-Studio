#!/bin/bash

# ==============================================================================
# Accelerate 启动参数
# ==============================================================================
NUM_GPUS=8
MIXED_PRECISION="bf16" # fp16, bf16, no

# ==============================================================================
# 训练脚本参数 (快速填写)
# ==============================================================================

# 1. 数据集配置
BASE_PATH="/mnt/raid0/docker-workspace/danbooru/1_2024"
METADATA_PATH=""
DATA_DIR="/mnt/raid0/docker-workspace/danbooru/1_2024" # 仅在 --use_image_text_pairs 时使用
USE_PAIRS=true # false: 使用 metadata 文件, true: 使用图片+txt目录
RECURSIVE=true  # 是否递归加载子文件夹 (仅对 USE_PAIRS=true 有效)
PREBUCKET_INDEX="/mnt/raid0/linux-train/diffusion-model-v1/dan2024bucket/prebucket_index_512.jsonl" # 可选: 预分桶索引 jsonl 路径
JSONL_INDEX=""     # 可选: jsonl metadata 行偏移索引路径

# 2. 训练模式
TRAIN_OMNI=true # 是否开启 Omni/编辑模式
USE_ALPHA_VAE=false

# 3. 模型文件路径
QWEN_MODEL="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b/model.safetensors"
FLUX_VAE="/mnt/raid0/linux-train/diffusion-model-v1/flux2-vae/diffusion_pytorch_model.safetensors"
QWEN_TOKENIZER="/mnt/raid0/linux-train/diffusion-model-v1/qwen3.5-2b"
QWEN_MODEL_TYPE="qwen3_5"
QWEN_MODEL_SIZE="2B"
SIGLIP_MODEL="" # 可选

# 4. 训练超参数
BATCH_SIZE=32
GRAD_ACCUM=2          # 梯度累积步数
LR=1e-4
EPOCHS=100
SAVE_STEPS=2000       # 每多少个 step 保存一次模型
NUM_WORKERS=8         # DataLoader 线程数 (训练时读取数据)
RESOLUTION="512 512"
MAX_BUCKET=1024
OUTPUT_DIR="models/Complextro/gemma"

# 5. DiT 模型架构 (对应脚本中注释掉的参数)
# 默认 2.25B 配置: num_layers=10, hidden_size=2304, heads=24, head_dim=96
# 备选 8层 配置: num_layers=8, hidden_size=3072, heads=24, head_dim=128
NUM_LAYERS=20
HIDDEN_SIZE=2304
NUM_HEADS=24
HEAD_DIM=96

# 6. 采样预览配置
SAMPLE_PROMPTS=(
    "1girl, solo, long hair, breasts, blue eyes, shirt, black hair, hair ornament, closed mouth, cleavage, medium breasts, white shirt, upper body, white hair, multicolored hair, outdoors, open clothes, sky, glasses, two-tone hair, bra, from side, coat, fur trim, profile, expressionless, feathers, black bra, black coat, open coat, sunset, looking ahead, feather hair ornament, looking afar, fur-trimmed coat, power lines, mountainous horizon, twilight, utility pole, evening, gradient sky, dusk"
    "1girl, solo, long hair, breasts, looking at viewer, blue eyes, blonde hair, navel, jewelry, sitting, medium breasts, collarbone, earrings, parted lips, necklace, blurry, bracelet, lips, kneeling, from above, looking up, between breasts, circlet, realistic, pasties, carpet, head chain"
    "1girl, solo, short hair, dress, holding, upper body, white hair, short sleeves, outdoors, wings, sky, puffy sleeves, white dress, water, puffy short sleeves, ocean, animal, bob cut, cat, feathered wings, angel wings, white wings, horizon, angel, holding animal, black cat, white cat, holding cat,"
    "seia \(blue archive\), 1girl, solo, long hair, breasts, looking at viewer, blonde hair, animal ears, jacket, tail, yellow eyes, swimsuit, ponytail, outdoors, small breasts, parted lips, open clothes, sky, alternate costume, choker, day, cloud, armpits, open jacket, halo, collar, arms up, blue sky, animal ear fluff, parted bangs, clothing cutout, fox ears, one-piece swimsuit, bare legs, covered navel, fox tail, highleg, fox girl, extra ears, arms behind head, visor cap, white one-piece swimsuit, highleg swimsuit, yellow jacket, casual one-piece swimsuit, red collar, yellow halo, yellow headwear"
)

# ==============================================================================
# 启动命令
# ==============================================================================

CMD="accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=$MIXED_PRECISION \
    train_complextro_ddp.py \
    --base_path $BASE_PATH \
    --data_dir $DATA_DIR \
    --qwen_model_file $QWEN_MODEL \
    --qwen_model_type $QWEN_MODEL_TYPE \
    --qwen_model_size $QWEN_MODEL_SIZE \
    --vae_file $FLUX_VAE \
    --qwen_tokenizer_dir $QWEN_TOKENIZER \
    --siglip_model_file \"$SIGLIP_MODEL\" \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --num_workers $NUM_WORKERS \
    --num_epochs $EPOCHS \
    --save_steps $SAVE_STEPS \
    --train_resolution $RESOLUTION \
    --max_bucket_reso $MAX_BUCKET \
    --output_dir $OUTPUT_DIR \
    --num_layers $NUM_LAYERS \
    --hidden_size $HIDDEN_SIZE \
    --num_attention_heads $NUM_HEADS \
    --attention_head_dim $HEAD_DIM"

# 动态拼接 SAMPLE_PROMPTS
if [ ${#SAMPLE_PROMPTS[@]} -gt 0 ]; then
    CMD="$CMD --sample_prompts"
    for prompt in "${SAMPLE_PROMPTS[@]}"; do
        CMD="$CMD \"$prompt\""
    done
fi

if [ "$USE_PAIRS" = false ] && [ -n "$METADATA_PATH" ]; then
    CMD="$CMD --metadata_path $METADATA_PATH"
fi

if [ "$USE_PAIRS" = true ]; then
    CMD="$CMD --use_image_text_pairs"
    if [ "$RECURSIVE" = true ]; then
        CMD="$CMD --recursive"
    fi
fi

if [ "$TRAIN_OMNI" = true ]; then
    CMD="$CMD --train_omni"
fi

if [ "$USE_ALPHA_VAE" = true ]; then
    CMD="$CMD --use_alpha_layer_vae"
fi

if [ -n "$PREBUCKET_INDEX" ]; then
    CMD="$CMD --prebucket_index_path $PREBUCKET_INDEX"
fi

if [ -n "$JSONL_INDEX" ]; then
    CMD="$CMD --jsonl_index_path $JSONL_INDEX"
fi

echo "执行命令: $CMD"
eval $CMD
