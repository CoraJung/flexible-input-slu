#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=fluent_best
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=ATS_1_FSC_%j.out
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
EXPERIMENT=experiments.experiment_triplet_combinedsystem.ExperimentRunnerTriplet

# Execute the script

# Regular Bert Encoder, train on GT+ASR, uw on audio, test on GT with combined system, our cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gtasr_gt --model-dir=$MODEL_DIR/gtasr_ours/uw \
--experiment=$EXPERIMENT --finetune-bert --unfreezing-type=1 > $MODEL_DIR/gtasr_ours/uw/gt_combinedsys.out

# Regular Bert Encoder, train on GT+ASR, uw on audio, test on ASR with combined system, our cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gtasr_asr --model-dir=$MODEL_DIR/gtasr_ours/uw \
--experiment=$EXPERIMENT --finetune-bert --unfreezing-type=1 --infer-only > $MODEL_DIR/gtasr_ours/uw/asr_combinedsys.out

# And we're done!
