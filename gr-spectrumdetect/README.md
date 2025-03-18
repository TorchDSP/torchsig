## GR-Spectrumdetect Overview
---
gr-spectrumdetect is an open-source example of using a trained model from TorchSig Wideband with GNU Radio for RF energy detection. 

`detect.pt` can be downloaded with the bash `trained_model_download.sh` script in `gr-spectrumdetect/examples/`.
- If `trained_model_download.sh` does not work, try using `trained_model_download_github.sh` (Downloads the model from [TorchSig's Github v0.6.0 Release Notes](https://github.com/TorchDSP/torchsig/releases/tag/v0.6.0))

`detect.pt` is a YOLOv8x model trained for detection with the following settings:
- single_cls=True
- 1024x1024 spectrograms
- gray scale black hot images
- Wideband with level 2 impairments with no signal overlap 
    - `torchsig/datasets/conf.py` -> `WidebandImpairedTrainConfig` -> `overlap_prob = 0.0`

### Notes     
- The first class of `wideband_yolo.yaml` has been modified to say `signal` because this training method is detection only. 

### Training
Training YOLOv8x began with the pretrained YOLOv8x model on COCO and runs for one epoch. The first layer was frozen and the learning rate lr0 was set to 0.0033329 and optimizer set to SGD. `detect.pt` was trained with the below command:

```
cd examples/

yolo detect train data=wideband_yolo.yaml model=yolov8x pretrained=yolov8x.pt device=0 epochs=1 batch=32 save=True save_period=1 single_cls=True imgsz=1024 name=8x_freeze1 cos_lr=False cache=False workers=16 freeze=1 lr0=0.0033329 optimizer=SGD
```


## Installation with Docker
---
The following command downloads a couple files needed to run the GNU Radio block:
```
$ cd torchsig/gr-spectrumdetect/examples
$ bash trained_model_download.sh
```
Now move back to the `gr-spectrumdetect` directory, then install and run the docker container:
```
$ cd ../gr-spectrumdetect
$ bash build_docker.sh
$ xhost +
$ bash run_docker.sh
```
You are now running inside the docker:
```
# source  /opt/gnuradio/v3.10/setup_env.sh
cd /build/gr-spectrumdetect/examples/
gnuradio-companion example.grc &
```

## Installation without Docker
---
Clone the `torchsig` repository and install using the following commands:
```
git clone https://github.com/TorchDSP/torchsig.git
cd torchsig
pip install .
cd gr-spectrumdetect
mkdir build
cd build
cmake ../
make install
cd ../examples/
bash trained_model_download.sh
gnuradio-companion example.grc &
```

## Generating the Datasets and training with Command Line
```
cd torchsig/gr-spectrumdetect/examples
bash generate.sh
bash make_yolo.sh
python3 verify_yolo_dataset_plot.py
bash train.sh
```

## License
---
gr-spectrumdetect is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. TorchSig has no connection to MIT, other than through the use of this license.
