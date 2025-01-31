#!/bin/bash


destination_path=detect.pt
download_url=https://bucket.ltsnet.net/torchsig/models/detect.pt

curl -L -o "$destination_path" "$download_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi

destination_path=11s.pt
download_url=https://bucket.ltsnet.net/torchsig/models/11s.pt

curl -L -o "$destination_path" "$download_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi

destination_path=xcit.ckpt
download_url=https://bucket.ltsnet.net/torchsig/models/xcit.ckpt

curl -L -o "$destination_path" "$download_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi
