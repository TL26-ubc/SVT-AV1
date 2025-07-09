#!/bin/bash
# take first argument as video source path files
video_path="$1"

# check if video_path is provided
if [ -z "$video_path" ]; then
  echo "Usage: $0 <video_path>"
  exit 1
fi

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
  # Skip empty lines
  if [ -n "$line" ]; then
    # Get the filename from the URL (everything after the last slash)
    filename=$(basename "$line")

    # Check if the file already exists
    if [ -f "$filename" ]; then
      echo "Skipping download — '$filename' already exists."
    else
      echo "Downloading video from: $line"
      wget "$line"
    fi
  else
    echo "Skipping empty line."
  fi
done < "$video_path"
