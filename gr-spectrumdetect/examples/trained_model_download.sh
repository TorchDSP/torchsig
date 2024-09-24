#!/bin/bash


destination_path=detect.pt

file_id=1vdzNyXjnZ61D2vruFhslerNscL3rAV8c
file_url="https://drive.usercontent.google.com/download?id=$file_id&export=download"

confirmation_page=$(curl -s -L "$file_url")

file_id=$(echo "$confirmation_page" | grep -oE "name=\"id\" value=\"[^\"]+" | sed 's/name="id" value="//')
file_confirm=$(echo "$confirmation_page" | grep -oE "name=\"confirm\" value=\"[^\"]+" | sed 's/name="confirm" value="//')
file_uuid=$(echo "$confirmation_page" | grep -oE "name=\"uuid\" value=\"[^\"]+" | sed 's/name="uuid" value="//')

download_url="https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=$file_confirm&uuid=$file_uuid"

curl -L -o "$destination_path" "$download_url"

if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
fi
