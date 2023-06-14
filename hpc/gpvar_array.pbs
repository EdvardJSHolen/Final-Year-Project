#!/bin/bash
#PBS -l walltime=36:00:00
#PBS -l select=1:ncpus=4:mem=24gb
#PBS -N gpexchange
#PBS -J 0-24

export NUM_PROCESSES="25"
export CONFIG_PATH="configs/gpvar_0.json" 

module load anaconda3/personal

cd $HOME/Final-Year-Project/gpvar/training_scripts
poetry shell
poetry update
poetry run python exchange.py