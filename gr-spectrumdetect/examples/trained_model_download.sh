#!/bin/bash


destination_path=detect.pt
download_url=https://bucket.ltsnet.net/torchsig/models/detect.pt

curl -L -o "$destination_path" "$download_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi
