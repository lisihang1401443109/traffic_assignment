#!/bin/bash
#
#SBATCH --job-name=traffic_greedy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --output=traffic-%j.out
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sli13@scu.edu

module load Anaconda3
conda activate traffic
python /WAVE/users/unix/sli13/workspace/traffic_assignment/test.py