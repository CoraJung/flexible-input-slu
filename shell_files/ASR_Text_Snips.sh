#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=ASR_Text_Snips
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=ASR_Text_Snips_%j.out
#SBATCH --gres=gpu:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATASET=snips
DATA_PATH={YOUR_PATH}/snips_slu/data
MODEL_DIR={YOUR_PATH}/best_chkpt/snips

# Execute the script
echo "Model 3.1 - Finetuning BERT - Train on both Raw+ASR - Our config - Test on ASR"
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gtasr_asr \
--model-dir=$MODEL_DIR/bert_trainboth_testasr --finetune-bert > ASR_Text_Snips_testasr.out

echo "Model 3.2 - Finetuning BERT - Train on both Raw+ASR - Our config - Test on RAW"
python train.py --dataset=$DATASET --data-path=$DATA_PATH/gtasr_gt \
--model-dir=$MODEL_DIR/bert_trainboth_testraw --finetune-bert --infer-only > ASR_Text_Snips_testraw.out
