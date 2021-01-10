#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=fluent_train
#SBATCH --mail-type=END
#SBATCH --mail-user=sjc433@nyu.edu
#SBATCH --output=fluent_train_%j.out
#SBATCH --gres=gpu:v100:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Activate the conda environment
module load anaconda3
source activate s2i_env
pip install -r requirements.txt

# Execute the script
python train.py 

# And we're done!
