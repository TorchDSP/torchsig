# TorchSig Examples
This folder contains sample Jupyter Notebooks that demonstrate some of the capabilities of TorchSig. Some of the later examples depend on the earlier ones being run first, for example in order to generate a dataset, therefore it is recommended to review them in order.

| Notebook | Description  |
| -------- | -----------  |
| 00_example_narrowband_dataset.ipynb | Create the full narrowband dataset with all signals, saves it to disk and plots the results |
| 01_example_modulations_dataset.ipynb | Demonstrates how to create a custom narrowband dataset by selecting a subset of signals |
| 02_example_narrowband_classifier.ipynb | Trains a narrowband classifier model from the narrowband dataset saved to disk |
| 03_example_wideband_dataset.ipynb | Create the full wideband dataset with all signals, saves it to disk and plots the results |
| 04_example_wideband_modulations_dataset.ipynb | Demonstrates how to create a custom wideband dataset by selecting a subset of signals |
| 05_example_wideband_yolo_to_disk.ipynb | Trains a wideband YOLOv8 classifier model from the wideband dataset saved to disk |
| 06_example_wideband_yolo.ipynb | More details on wideband YOLOv8 training and runs the model against a test dataset for inference |
| 07_example_narrowband_yolo.ipynb | More details on narrowband YOLOv8 training and runs the model against a test dataset for inference |
| 08_example_optuna_yolo.ipynb |  Optimizes wideband model hyperparameters using Optuna |
| 09_example_synthetic_spectrogram_dataset.ipynb | Building synthetic spectrograms using context free grammar (CFG) |
| 10_example_yolo_annotation_tool.ipynb | Demonstration of YOLO annotation tool for selecting bounding boxes for detections |
| 11_example_timm_models.ipynb | Loading models using timm |
| 12_example_sigmf.ipynb | How to read from SigMF files into TorchSig as a dataset |
| 13_example_pretrained_models.ipynb | How to use our pretrained models on Narrowband and Wideband |
