#!/bin/bash 

# Script to run python code on HPC clusters, must be run from within the hpc folder. Input file can be in any folder

filename=$(basename "$1")
filedir=$(dirname "$1")
PBS_STR=$'
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000 

module load anaconda3/personal

cd $PBS_O_WORKDIR
cd ..
cd FILEDIR
poetry update
poetry run python FILENAME
'

PBS_STR=${PBS_STR//FILENAME/$filename}
PBS_STR=${PBS_STR//FILEDIR/$filedir}

cd log-files
rm tmp.pbs
echo "$PBS_STR" > tmp.pbs

qsub tmp.pbs

rm tmp.pbs

