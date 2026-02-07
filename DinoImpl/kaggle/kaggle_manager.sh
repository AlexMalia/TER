#!/bin/bash
# Kaggle Manager Script for DINO Training

set -e

KAGGLE_USERNAME="alexdelaveau"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CODE_DATASET_DIR="$SCRIPT_DIR/code-dataset"
CHECKPOINT_DATASET_DIR="$SCRIPT_DIR/checkpoint-dataset"
KERNEL_DIR="$SCRIPT_DIR/kernel"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

KAGGLE_CMD="uv run kaggle"

check_kaggle_cli() {
    if ! $KAGGLE_CMD --version &> /dev/null; then
        echo "Error: Kaggle CLI not installed. Run: uv add kaggle"
        exit 1
    fi

    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo "Error: Kaggle API credentials not found."
        echo "1. Go to kaggle.com -> Account -> Create New API Token"
        echo "2. Place kaggle.json in ~/.kaggle/"
        echo "3. Run: chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi
}

update_username() {
    if [ "$KAGGLE_USERNAME" == "YOUR_KAGGLE_USERNAME" ]; then
        echo "Error: Please edit this script and set your KAGGLE_USERNAME"
        exit 1
    fi

    sed -i "s/YOUR_KAGGLE_USERNAME/$KAGGLE_USERNAME/g" "$CODE_DATASET_DIR/dataset-metadata.json" 2>/dev/null || true
    sed -i "s/YOUR_KAGGLE_USERNAME/$KAGGLE_USERNAME/g" "$CHECKPOINT_DATASET_DIR/dataset-metadata.json" 2>/dev/null || true
    sed -i "s/YOUR_KAGGLE_USERNAME/$KAGGLE_USERNAME/g" "$KERNEL_DIR/kernel-metadata.json" 2>/dev/null || true
}

prepare_code_dataset() {
    echo "Preparing code dataset..."
    
    # Nettoyer les anciens fichiers
    rm -rf "$CODE_DATASET_DIR/src" "$CODE_DATASET_DIR/configs" "$CODE_DATASET_DIR/scripts"
    rm -rf "$CODE_DATASET_DIR/.venv" "$CODE_DATASET_DIR/.venv.zip"
    
    # Copier les nouveaux fichiers
    cp -r "$PROJECT_ROOT/src" "$CODE_DATASET_DIR/"
    cp -r "$PROJECT_ROOT/configs" "$CODE_DATASET_DIR/"
    cp -r "$PROJECT_ROOT/scripts" "$CODE_DATASET_DIR/"
    cp "$PROJECT_ROOT/pyproject.toml" "$CODE_DATASET_DIR/"
    cp "$PROJECT_ROOT/uv.lock" "$CODE_DATASET_DIR/"

    # Créer .kaggleignore
    cat > "$CODE_DATASET_DIR/.kaggleignore" << 'EOF'
.venv/
.venv.zip
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.git/
.DS_Store
EOF
    
    echo "Code dataset prepared at: $CODE_DATASET_DIR"
}

cmd_setup() {
    echo "============================================"
    echo "Setting up Kaggle datasets and kernel"
    echo "============================================"

    check_kaggle_cli
    update_username
    prepare_code_dataset

    echo ""
    echo "Creating code dataset on Kaggle..."
    $KAGGLE_CMD datasets create -p "$CODE_DATASET_DIR" --dir-mode zip

    echo ""
    echo "Creating checkpoint dataset on Kaggle..."
    # Create a placeholder file so the dataset isn't empty
    touch "$CHECKPOINT_DATASET_DIR/.gitkeep"
    $KAGGLE_CMD datasets create -p "$CHECKPOINT_DATASET_DIR" --dir-mode zip

    echo ""
    echo "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run './kaggle/kaggle_manager.sh run' to start training"
    echo "  2. Run './kaggle/kaggle_manager.sh status' to check progress"
    echo "  3. Run './kaggle/kaggle_manager.sh output' to download results"
}

cmd_push_code() {
    echo "============================================"
    echo "Updating code dataset on Kaggle"
    echo "============================================"

    check_kaggle_cli
    prepare_code_dataset

    # NE PAS faire cd "$CODE_DATASET_DIR"
    $KAGGLE_CMD datasets version -p "$CODE_DATASET_DIR" -m "Code update $(date +%Y-%m-%d_%H:%M)" --dir-mode zip

    echo "Code dataset updated!"
}

cmd_run() {
    echo "============================================"
    echo "Pushing and running training kernel"
    echo "============================================"

    check_kaggle_cli
    cmd_push_code

    echo ""
    echo "Waiting 30s for dataset to propagate on Kaggle..."
    sleep 30

    echo "Pushing kernel to Kaggle..."
    # Celui-ci peut rester avec cd car kernel/ ne devrait pas créer de venv
    cd "$KERNEL_DIR"
    $KAGGLE_CMD kernels push -p .

    echo ""
    echo "Kernel submitted!"
    echo "View at: https://www.kaggle.com/$KAGGLE_USERNAME/dino-training"
    echo ""
    echo "Run './kaggle/kaggle_manager.sh status' to check progress"
}

cmd_status() {
    check_kaggle_cli
    $KAGGLE_CMD kernels status "$KAGGLE_USERNAME/dino-training"
}

cmd_output() {
    check_kaggle_cli
    mkdir -p "$OUTPUT_DIR"
    cd "$OUTPUT_DIR"
    $KAGGLE_CMD kernels output "$KAGGLE_USERNAME/dino-training" -p .
    echo "Outputs downloaded to: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
}

cmd_logs() {
    xdg-open "https://www.kaggle.com/$KAGGLE_USERNAME/dino-training" 2>/dev/null || \
    open "https://www.kaggle.com/$KAGGLE_USERNAME/dino-training" 2>/dev/null || \
    echo "Open: https://www.kaggle.com/$KAGGLE_USERNAME/dino-training"
}

cmd_push_checkpoint() {
    echo "============================================"
    echo "Pushing checkpoint to Kaggle dataset"
    echo "============================================"

    check_kaggle_cli

    # Check if outputs have been downloaded
    local checkpoint_file="$OUTPUT_DIR/checkpoints/checkpoint_latest.pth"
    local history_file="$OUTPUT_DIR/checkpoints/history_latest.json"

    if [ ! -f "$checkpoint_file" ]; then
        echo "No checkpoint found in outputs. Downloading outputs first..."
        cmd_output
    fi

    if [ ! -f "$checkpoint_file" ]; then
        echo "Error: No checkpoint_latest.pth found in $OUTPUT_DIR/checkpoints/"
        echo "Make sure training has completed and produced a checkpoint."
        exit 1
    fi

    # Copy checkpoint files to the dataset root (--dir-mode zip flattens structure)
    cp "$checkpoint_file" "$CHECKPOINT_DATASET_DIR/"

    if [ -f "$history_file" ]; then
        cp "$history_file" "$CHECKPOINT_DATASET_DIR/"
    fi

    echo "Pushing checkpoint dataset to Kaggle..."
    # Try version update first; if dataset doesn't exist yet, create it
    if ! $KAGGLE_CMD datasets version -p "$CHECKPOINT_DATASET_DIR" -m "Checkpoint update $(date +%Y-%m-%d_%H:%M)" --dir-mode zip 2>/dev/null; then
        echo "Dataset doesn't exist yet, creating it..."
        $KAGGLE_CMD datasets create -p "$CHECKPOINT_DATASET_DIR" --dir-mode zip
    fi

    echo ""
    echo "Checkpoint pushed! Next run will auto-resume from this checkpoint."
}

cmd_help() {
    echo "Kaggle Manager for DINO Training"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup             - Initial setup: create datasets and kernel on Kaggle"
    echo "  push-code         - Upload/update code dataset to Kaggle"
    echo "  run               - Push code and run the training kernel"
    echo "  status            - Check kernel execution status"
    echo "  output            - Download training outputs"
    echo "  push-checkpoint   - Push latest checkpoint for resume on next run"
    echo "  logs              - Open kernel page in browser"
}

case "${1:-help}" in
    setup)            cmd_setup ;;
    push-code)        cmd_push_code ;;
    run)              cmd_run ;;
    status)           cmd_status ;;
    output)           cmd_output ;;
    push-checkpoint)  cmd_push_checkpoint ;;
    logs)             cmd_logs ;;
    help|*)           cmd_help ;;
esac