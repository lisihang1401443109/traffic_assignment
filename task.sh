#!/bin/bash
#
#SBATCH --job-name=traffic_greedy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --output=traffic-%j.out
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sli13@scu.edu

module load Anaconda3
conda activate traf
python /WAVE/users/unix/sli13/workspace/traffic_assignment/test.py > log_11_15
