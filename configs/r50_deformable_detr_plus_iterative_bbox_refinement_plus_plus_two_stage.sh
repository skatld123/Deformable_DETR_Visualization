#!/usr/bin/env bash
# GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage.sh
set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --epoch 100 \
    ${PY_ARGS}
