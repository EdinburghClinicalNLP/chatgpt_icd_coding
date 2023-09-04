#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --mem=32000  # memory in Mb
#SBATCH --partition=ILCC_CPU
#SBATCH -t 1-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=16

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

echo "Setting up bash enviroment"
source ~/.bashrc
set -e
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
PROJECT_NAME="chatgpt_icd_coding"
echo "Activating conda environment: ${PROJECT_NAME}"
conda activate ${PROJECT_NAME}

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
src_path=/home/${USER}/${PROJECT_NAME}/data/
dest_path=${SCRATCH_HOME}/${PROJECT_NAME}/data/
mkdir -p ${dest_path}  # make it if required
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
src_path=/home/${USER}/${PROJECT_NAME}/outputs/
dest_path=${SCRATCH_HOME}/${PROJECT_NAME}/outputs/
mkdir -p ${dest_path}  # make it if required
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Running experiment"
python scripts/evaluate.py --predictions_dir outputs/2023_08_25__09_19_37/predictions --groundtruth_path data/disch_raw_test_split.csv

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"