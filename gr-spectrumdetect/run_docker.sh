#!/bin/bash
docker run --privileged -v /etc/udev/rules.d/:/etc/udev/rules.d/ -it --rm --gpus all --network=host -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority -v /:/workspace/ --name grspectrumdetect -t grspectrumdetect:v01 /bin/bash

