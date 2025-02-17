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

LR0=0.000000001
MLP_LATENT=32,32
MLP_LAYERS=2,2
GNN_LAYERS=5,5
NUM_FEATURES=16,16 # TODO: currently not used 
MODE=classification
EPOCHS=1
BATCH_SIZE=1
# BATCH_SIZE=5
DATA_THREADS=10

MODEL_DIR="checkpoints" 
mkdir -p $MODEL_DIR
MODEL_PATH="$MODEL_DIR/model0.ckpt"

INFERENCE_OUT="${TEST_DATASET}-inference-2.out"

# default architecture
# gnn_model_protein_core0: [32, 23, 15, 32]
# gnn_model_protein_core1: [42, 34, 26, 34, 42]
# gnn_model_protein_core2: [53, 45, 37, 29, 45, 53]
# gnn_model_protein_core3: [64, 55, 47, 39, 47, 55, 64]
# gnn_model_ligand_core0: [16, 11, 7, 16]
# gnn_model_ligand_core1: [18, 13, 8, 18]
# gnn_model_ligand_core2: [20, 14, 9, 20]
# gnn_model_ligand_core3: [22, 17, 13, 17, 22]
# gnn_model_ligand_core4: [25, 20, 15, 20, 25]
# gnn_model_ligand_core5: [27, 22, 18, 14, 22, 27]
# gnn_model_ligand_core6: [29, 24, 20, 15, 24, 29]
# gnn_model_ligand_core7: [32, 27, 23, 19, 23, 27, 32]

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
    --mlp_latent ${MLP_LATENT} \
    --mlp_layers ${MLP_LAYERS} \
    --gnn_layers ${GNN_LAYERS} \
    --data_threads ${DATA_THREADS}\
    --epochs ${EPOCHS} \
    --inference_out ${INFERENCE_OUT} #\
    # --inference_only True