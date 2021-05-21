#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=ASR-Text-Speech-2-Snips
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=ATS_2_Snips_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATA_PATH={YOUR_PATH}/snips_slu/data
DATASET=snips
BERT_DIR={YOUR_PATH}/best_chkpt/snips/bert_trainboth_ours
FROZEN_MODEL_DIR={YOUR_PATH}/best_chkpt/snips/pretrain_bert/bert_frozen
EXPERIMENT=experiments.experiment_triplet_combinedsystem.ExperimentRunnerTriplet
# SCHEDULER=cycle

# Execute the script

# Load model3 pretrained on GT+ASR frozen, train on GT, ua on audio, test on GT, default cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_gt --experiment=$EXPERIMENT --model-dir=$FROZEN_MODEL_DIR/gt_ours/ua \
--unfreezing-type=2 --bert-dir=$BERT_DIR > $FROZEN_MODEL_DIR/gt_ours/ua/gt_combinedsys.out

python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_asr --experiment=$EXPERIMENT --model-dir=$FROZEN_MODEL_DIR/gt_ours/ua \
--unfreezing-type=2 --bert-dir=$BERT_DIR --infer-only > $FROZEN_MODEL_DIR/gt_ours/ua/asr_combinedsys.out


# And we're done!
