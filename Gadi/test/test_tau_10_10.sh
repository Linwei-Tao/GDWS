#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -l jobfs=200GB
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd


projectDIR="GDWS"

module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/$PROJECT/$USER/$projectDIR
python3 GDWS.py --data_path=../datasets/cifar10 --dataset=cifar10 --save_dir=temp_dir --rand_seed=1 --tau_max=10 --tau_min=10
