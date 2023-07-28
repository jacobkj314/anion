#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=jacob.k.johnson@utah.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -o filename-%j

source ~/miniconda3/etc/profile.d/conda.sh

conda activate 38b

wandb enabled
export TRANSFORMER_CACHE="../../../cache"

make=allenai/
model=unifiedqa-v2-t5-large-1251000
use_deepspeed=false
is_checkpoint=false

python dataBundler.py -fa #-use_test #to get test results instead of dev results
echo BEGINNING RUN_SING ; bash run_single_unifiedqa.sh $make $model ../../../out/ $use_deepspeed $is_checkpoint; echo COMPLETED RUN_SING
echo BEGINNING COMPUTE_STATS ; bash compute_unifiedqa_stats.sh $model ../../../out/ $use_deepspeed ; echo COMPLETED COMPUTE_STATS 
