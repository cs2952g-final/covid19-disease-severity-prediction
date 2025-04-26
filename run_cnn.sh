#!/bin/bash

#SBATCH -J run_proj     	        # name
#SBATCH -N 1 						# all cores are on one node
#SBATCH -n 1                        # cores
#SBATCH -t 2-00:00 			        # time 5hrs per job days	
#SBATCH --mem 50G 				    # memory
#SBATCH -p bigmem


# bash file to run proj commands 

module load python
source ~/venv/dl_in_genomics_finalproj/bin/activate

python cnn_hannah.py 