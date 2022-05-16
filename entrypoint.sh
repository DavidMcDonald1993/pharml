#!/bin/bash

TRAIN_MAP=data/map/dataset.map
TEST_MAP=data/map/dataset.map # same map file for both is fine for inference?

# taken from runit_inference.sh
LR0=0.000000001 # not required for inference
MLP_LATENT=32,32
MLP_LAYERS=2,2
GNN_LAYERS=5,5
NUM_FEATURES=16,16
MODE=classification
EPOCHS=1
DATA_THREADS=4
BATCH_SIZE=8

# load pre-trained
MODEL_DIR="/pharML-Bind/pretrained-models/mh-gnnx5-ensemble/model_0/checkpoints" 
# mkdir -p $MODEL_DIR
MODEL_PATH="$MODEL_DIR/model0.ckpt"

# bind mount out
INFERENCE_OUT="/results/inference.out"

python /pharML-Bind/mldock_gnn.py \
    --map_train ${TRAIN_MAP} \
    --map_test ${TEST_MAP} \
    --mode ${MODE} \
    --restore ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --batch_size_test ${BATCH_SIZE} \
    --data_threads ${DATA_THREADS} \
    --epochs ${EPOCHS} \
    --inference_out ${INFERENCE_OUT} \
    --inference_only True \
    --mlp_latent ${MLP_LATENT} \
    --mlp_layers ${MLP_LAYERS} \
    --gnn_layers ${GNN_LAYERS} \
    --num_features ${NUM_FEATURES} 