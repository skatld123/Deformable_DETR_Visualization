#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --epoch 100 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}

# python -u main.py \
#     --resume /root/Deformable-DETR/exps/test1/r50_deformable_detr/checkpoint0054.pth \
#     --start_epoch 55 \
#     --epoch 100 \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}

# python main.py   --dataset_file coco   --coco_path /root/detr/dataset_clp/\   --output_dir outputs   --resume detr-r50_no-class-head.pth   
# --num_classes 2   --lr 1e-5   --lr_backbone 1e-6   --epochs 100   --num_workers=16
# python -u main.py --resume /root/Deformable-DETR/exps/test1/r50_deformable_detr/checkpoint0054.pth --output_dir exps/r50_deformable_detr