ROOT_DIR="$(pwd)/rlds_dataset"

for TASK_DIR in "${ROOT_DIR}"/robocasa_*; do
  [[ -d "${TASK_DIR}" ]] || continue

    TASK_NAME=$(basename "${TASK_DIR}")
    echo "Building dataset for ${TASK_NAME}..."

    cd "${TASK_DIR}"

    tfds build

done
echo "All datasets built successfully."