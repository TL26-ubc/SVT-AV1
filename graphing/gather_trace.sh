#!/bin/bash

# take the first argument as the file path
path="$1"
if [ -z "$path" ]; then
    echo "Usage: $0 <path_to_video_file>"
    exit 1
fi

output_console="$2"
if [ -z "$output_console" ]; then
    output_console="console2.log"
fi

iterations="$3"
if [ -z "$iterations" ]; then
    iterations=100000
fi

python3 ../python-gym/src/pyencoder/environment/train_refactor.py \
    --file "$path" \
    --output_dir ../Output \
    --wandb True \
    --total_iteration $iterations 2> >(tee $output_console >&2)