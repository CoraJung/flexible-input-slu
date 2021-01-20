#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=snips_train
#SBATCH --mail-type=END
#SBATCH --mail-user=wh916@nyu.edu
#SBATCH --output=snips_train_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate alexa_env
DATA_PATH=//misc/vlgscratch5/PichenyGroup/s2i-common/alexa-slu/snips_slu/
DATASET=snips

# Execute the script
python train.py --dataset=$DATASET --data_path=$DATA_PATH --enc-dim=512

# And we're done!
