path="$1"
if [ -z "$path" ]; then
    echo "Usage: $0 <path_to_video_file> [output_console] [iterations]"
    echo "Example: $0 /home/tl26/videos/football_cif.y4m console2.log 2"
    exit 1
fi

output_console="$2"
if [ -z "$output_console" ]; then
    output_console="console2.log"
fi

iterations="$3"
if [ -z "$iterations" ]; then
    iterations=2
fi

output_dir="$4"
if [ -z "$output_dir" ]; then
    output_dir="output_models"
    # check if the directory exists, if not create it
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
        echo "Created output directory: $output_dir"
    fi
fi

python av1env-training/src/train_refactor.py \
    --file "$path" \
    --output_dir "$output_dir" \
    --total_iteration "$iterations" \
    --wandb True \
    --batch_size 32 \
    2> >(tee $output_console >&2)