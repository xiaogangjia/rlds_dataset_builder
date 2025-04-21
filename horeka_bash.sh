#! /bin/bash
#SBATCH -p cpuonly # accelerated-h100 # dev_accelerated
#SBATCH -J robocasa_build

# Cluster Settings
#SBATCH -n 1       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 1200 # 30:00 # 2-00:00:00 # 1:00:00 # 2-00:00:00 ## # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --ntasks-per-node=1

# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-robolear/ll6323/rlds_logs/slurm_logs/%x_%j.out
#SBATCH --error=/home/hk-project-robolear/ll6323/rlds_logs/slurm_logs/%x_%j.err


# Activate the virtualenv / conda environment
source /home/hk-project-robolear/ll6323/miniconda3/bin/activate rlds_env

export TFDS_DATA_DIR="/hkfs/work/workspace/scratch/ll6323-david_dataset_2/rlds_robocasa"
ROOT_DIR="$(pwd)/rlds_dataset"

for TASK_DIR in "${ROOT_DIR}"/robocasa_*; do
  [[ -d "${TASK_DIR}" ]] || continue

    TASK_NAME=$(basename "${TASK_DIR}")
    echo "Building dataset for ${TASK_NAME}..."

    cd "${TASK_DIR}"

    tfds build

done
echo "All datasets built successfully."