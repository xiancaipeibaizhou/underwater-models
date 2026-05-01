#!/bin/bash

# ==============================================================================
# HTAN 诊断实验：强正则化保守疗法 (最简基线 vs 完整架构)
# ==============================================================================

EXP_TIME=$(date +"%Y%m%d_%H%M%S_Conservative")

echo "----------------------------------------------------------------------"
echo "🚨 启动保守疗法诊断实验 | 时间戳: ${EXP_TIME}"
echo "🚨 强约束条件: Patience=8, Weight_Decay=1e-3, Dropout=0.3"
echo "----------------------------------------------------------------------"

# echo "▶️ [1/2] 诊断 A：保守版 最简模型 (G0_P0_TE0_TA0)"
# python demo_light.py \
#     --exp_time ${EXP_TIME} \
#     --model HTAN \
#     --use_graph 0 --use_prior_mask 0 --use_temporal_encoder 0 --use_temporal_attention 0 \
#     --patience 8 \
#     --weight_decay 1e-3 \
#     --dropout 0.3

echo "▶️ [2/2] 诊断 B：保守版 完整架构 (G1_P1_TE1_TA1)"
python demo_light.py \
    --exp_time ${EXP_TIME} \
    --model HTAN \
    --use_graph 1 --use_prior_mask 1 --use_temporal_encoder 1 --use_temporal_attention 1 \
    --patience 20 \
    --weight_decay 1e-3 \
    --dropout 0.3

echo "✅ 诊断实验运行完毕！"



//蒸馏学习
# 1. 删掉或者移走旧的不匹配权重（防止等下加载串了）
rm -rf ./checkpoints/ssl_pretrain/*.ckpt

# 2. 重新启动自监督预训练（让新版知识图谱吸收 Teacher 的特征）
python pretrain.py

python finetune.py