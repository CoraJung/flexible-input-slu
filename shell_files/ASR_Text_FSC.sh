#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=ASR_Text_FSC
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=ASR_Text_FSC_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATA_PATH=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/fluentai_asr/data
MODEL_DIR=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/best_chkpt/fluentai

# Execute the script
echo "Model 3 - Finetuning BERT - Train on both GT+ASR - Our config - Test on GT"
python train.py --data-path=$DATA_PATH/gtasr_gt \
--model-dir=$MODEL_DIR/bert_trainboth_ours --finetune-bert > ASR_Text_FSC_testraw.out

echo "Model 3 - Finetuning BERT - Train on both GT+ASR - Our config - Test on ASR"
python train.py --data-path=$DATA_PATH/gtasr_asr \
--model-dir=$MODEL_DIR/bert_trainboth_ours --finetune-bert --infer-only > ASR_Text_FSC_testasr.out
