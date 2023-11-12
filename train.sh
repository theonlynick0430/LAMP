#!/bin/bash

#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --job-name="lamp"
#SBATCH --output=/home/niksrid/mental-models/LAMP/train.log
#SBATCH --mail-type=BEGIN,END,NONE,FAIL,REQUEUE

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate LAMP
cd /home/niksrid/mental-models/LAMP
python train_lamp.py --config configs/bridge.yaml
wait