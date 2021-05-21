#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=snipsBERT_ours
#SBATCH --mail-type=END
#SBATCH --mail-user=sjc433@nyu.edu
#SBATCH --output=snipsBERT_ours_all%j.out
#SBATCH --gres=gpu:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATASET=snips
DATA_PATH=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/snips_slu/
MODEL_DIR=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/snips

# Execute the script
echo "Model 1 - Not Finetuning BERT - our config"
python train.py --dataset=$DATASET --data-path=$DATA_PATH --model-dir=$MODEL_DIR/bert_frozen_ours > snipsBERT_frozen_ours.out

echo "Model 2 - Finetuning BERT - our config"
python train.py --dataset=$DATASET --data-path=$DATA_PATH --model-dir=$MODEL_DIR/bert_finetune_ours --finetune-bert > snipsBERT_finetune_ours.out

echo "Model 1 - Not Finetuning BERT - our config - inference on ASR"
python train.py --dataset=$DATASET --data-path=$DATA_PATH \
--model-dir=$MODEL_DIR/bert_frozen_ours --infer-only > snipsBERT_frozen_ours_infer_asr.out

echo "Model 2 - Finetuning BERT - our config - inference on ASR"
python train.py --dataset=$DATASET --data-path=$DATA_PATH \
--model-dir=$MODEL_DIR/bert_finetune_ours --finetune-bert --infer-only > snipsBERT_finetune_ours_infer_asr.out
