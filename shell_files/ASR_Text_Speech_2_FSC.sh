#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=fluent_best
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=ATS_2_FSC_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATA_PATH=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/fluentai_asr/data
DATASET=fsc
MODEL_DIR=/misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/fluentai/np_bert
BERT_DIR=/misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/fluentai/bert_trainboth_ours
FROZEN_MODEL_DIR=/misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/fluentai/pretrain_bert/bert_frozen
EXPERIMENT=experiments.experiment_triplet_combinedsystem.ExperimentRunnerTriplet

# Execute the script

# Load model3 pretrained on GT+ASR frozen, train on GT, ua on audio, test on GT using combined system, default cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_gt --experiment=$EXPERIMENT --model-dir=$FROZEN_MODEL_DIR/gt_ours/ua \
--unfreezing-type=2 --bert-dir=$BERT_DIR > $FROZEN_MODEL_DIR/gt_ours/ua/gt_combinedsys.out

# Load model3 pretrained on GT+ASR frozen, train on GT, ua on audio, test on ASR using combined system, default cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_asr --experiment=$EXPERIMENT --model-dir=$FROZEN_MODEL_DIR/gt_ours/ua \
--unfreezing-type=2 --bert-dir=$BERT_DIR --infer-only > $FROZEN_MODEL_DIR/gt_ours/ua/asr_combinedsys.out

# And we're done!
