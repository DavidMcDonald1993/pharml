#!/bin/bash

export LD_LIBRARY_PATH=${HOME}/miniconda/envs/pharml-env-cuda:${LD_LIBRARY_PATH}

TRAINING_MAP=data/bindingdbsubset/map/dataset.map
TEST_MAP=data/bindingdbtest/map/dataset.map

BATCH_SIZE=1
EPOCHS=5

MODEL_PATH="checkpoints/model0.ckpt"

INFERENCE_OUT="bindingdb-test-inference.out"

python pharML-Bind/mldock_gnn.py \
    --map_train ${TRAINING_MAP} \
    --map_test ${TEST_MAP} \
    --mode classification \
    --restore ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --batch_size_test ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --inference_out ${INFERENCE_OUT} \
    --inference_only TRUE