#!/bin/bash
#PBS -N constraint_dropping
#PBS -j oe
#PBS -o logs.out
#PBS -e logs.err
#PBS -lselect=1:ncpus=64:mem=98gb
#PBS -lwalltime=01:00:00

module load anaconda3/personal

cd $PBS_O_WORKDIR
conda env create -f environment.yml 
source activate constraint_dropping
python3 constraint_dropping/main.py
