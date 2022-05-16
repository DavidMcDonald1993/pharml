#!/bin/bash

#SBATCH --job-name=PHARML
#SBATCH --output=PHARML.out
#SBATCH --error=PHARML.err
#SBATCH --qos bbgpu
#SBATCH --account hesz01
#SBATCH --gres gpu:p100:1
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G

# module purge; module load bluebear
# module load TensorFlow/1.15.0-fosscuda-2019b-Python-3.7.4
# module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4

# pip install --user dm-sonnet==1.25\
#     graph-nets\
#     tensorflow-probability==0.7.0

# probably not needed on BlueBEAR? 
export LD_LIBRARY_PATH=${HOME}/miniconda/envs/pharml-env-cuda:${LD_LIBRARY_PATH}

TEST_DATASET=dude

TRAIN_MAP=data/bindingdbsubset/map/dataset.map
TEST_MAP=data/${TEST_DATASET}/map/dataset.map

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
MODEL_DIR="pharML-Bind/pretrained-models/mh-gnnx5-ensemble/model_0/checkpoints" 
# mkdir -p $MODEL_DIR
MODEL_PATH="$MODEL_DIR/model0.ckpt"

INFERENCE_OUT_DIR="output/pretrained"
mkdir -p $INFERENCE_OUT_DIR
INFERENCE_OUT="${INFERENCE_OUT_DIR}/pretrained-${TEST_DATASET}-inference-0.out"

# new architecture
# gnn_model_protein_core0: [32, 7]
# gnn_model_protein_core1: [40, 9]
# gnn_model_protein_core2: [48, 29, 48]
# gnn_model_protein_core3: [56, 41, 27, 56]
# gnn_model_protein_core4: [64, 51, 39, 51, 64]
# gnn_model_ligand_core0: [32, 7]
# gnn_model_ligand_core1: [40, 9]
# gnn_model_ligand_core2: [48, 29, 48]
# gnn_model_ligand_core3: [56, 41, 27, 56]
# gnn_model_ligand_core4: [64, 51, 39, 51, 64]

python pharML-Bind/mldock_gnn.py \
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