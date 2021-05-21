#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=fsc_text_speech
#SBATCH --mail-type=END
#SBATCH --mail-user=sjc433@nyu.edu
#SBATCH --output=delete_errorMessage_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATA_PATH=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/fluent_asr/data
DATASET=fsc
MODEL_DIR=/misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/fluentai
# SCHEDULER=cycle
EXPERIMENT=experiments.experiment_triplet_combinedsystem.ExperimentRunnerTriplet

# Execute the script
#### Combined ACC
# Regular Bert Encoder, train on GT, ua on audio, test on GT with combined system, our cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_gt --model-dir=$MODEL_DIR/np_bert/gt_ours/ua \
--experiment=$EXPERIMENT --finetune-bert --unfreezing-type=2 > $MODEL_DIR/np_bert/gt_ours/ua/gt_combinedsys.out

# Regular Bert Encoder, train on GT, ua on audio, test on ASR with combined system, our cfg
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gt_asr --model-dir=$MODEL_DIR/np_bert/gt_ours/ua \
--experiment=$EXPERIMENT --finetune-bert --unfreezing-type=2 --infer-only > $MODEL_DIR/np_bert/gt_ours/ua/asr_combinedsys.out

# And we're done!
