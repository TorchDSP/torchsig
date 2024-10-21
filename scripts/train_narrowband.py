#!/usr/bin/env python

"""
Script to train a model using the TorchSig Narrowband dataset.

This script allows for training a neural network model on the TorchSig Narrowband dataset with configurable options
such as model name, number of epochs, batch size, learning rate, and more.

Example usage:
    python train_narrowband.py --model_name xcit --num_epochs 10 --batch_size 64

Command line arguments:
    --data_path: Path to the dataset. Default is '../datasets/narrowband_test_QA'.
    --model_name: Name of the model to use for training. Default is 'xcit'.
    --num_epochs: Number of training epochs. Default is 2.
    --batch_size: Batch size for data loaders. Default is 32.
    --num_workers: Number of workers for data loading. Default is 16.
    --learning_rate: Learning rate for optimizer. Default is 1e-3.
    --input_channels: Number of input channels. Default is 2.
    --impaired: Include impaired signals in the dataset.
    --qa: Enable QA signals in the dataset.
    --checkpoint_path: Path to checkpoint to resume training.
    --use_datamodule: Use custom datamodule for data loading.
"""

import argparse

def main():
    """
    Main function to train a model using the TorchSig Narrowband dataset.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model using TorchSig Narrowband dataset.')
    parser.add_argument('--data_path', type=str, default='../datasets/narrowband_test_QA',
                        help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='xcit',
                        help='Model name to use for training')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--input_channels', type=int, default=2,
                        help='Number of input channels')
    parser.add_argument('--impaired', type=bool, default=True,
                        help='Include impaired signals')
    parser.add_argument('--qa', type=bool, default=True,
                        help='Enable QA signals')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--use_datamodule', type=bool, default=True,
                        help='Use custom datamodule')
    args = parser.parse_args()

    # TorchSig imports
    from torchsig.transforms.target_transforms import DescToClassIndex
    from torchsig.transforms.transforms import (
        RandomPhaseShift,
        Normalize,
        ComplexTo2D,
        Compose,
    )
    from torchsig.utils.narrowband_trainer import NarrowbandTrainer
    from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
    from torchsig.datasets.datamodules import NarrowbandDataModule
    import numpy as np
    import cv2
    import os
    import matplotlib.pyplot as plt

    # Get the list of class names from TorchSigNarrowband dataset
    class_list = list(TorchSigNarrowband._idx_to_name_dict.values())
    num_classes = len(class_list)

    # Specify data transformations to be applied to the dataset
    transform = Compose(
        [
            RandomPhaseShift(phase_offset=(-1, 1)),  # Randomly shift the phase of the signal
            Normalize(norm=np.inf),                  # Normalize the signal
            ComplexTo2D(),                           # Convert complex signal to 2D representation
        ]
    )

    # Specify target transformation (e.g., mapping description to class index)
    target_transform = DescToClassIndex(class_list=class_list)

    if args.use_datamodule:
        # Create the data module for the narrowband dataset
        datamodule = NarrowbandDataModule(
            root=args.data_path,             # Path to the dataset
            qa=args.qa,                      # Enable QA signals
            impaired=args.impaired,          # Include impaired signals
            transform=transform,             # Apply data transformations
            target_transform=target_transform,  # Apply target transformations
            batch_size=args.batch_size,         # Batch size for data loaders
            num_workers=args.num_workers,       # Number of workers for data loading
        )
        datamodule_param = datamodule
    else:
        datamodule_param = None

    # Initialize the trainer with desired parameters
    trainer = NarrowbandTrainer(
        model_name = args.model_name,             # Specify the model to use
        num_epochs = args.num_epochs,             # Number of training epochs
        batch_size = args.batch_size if not args.use_datamodule else None,
        num_workers = args.num_workers if not args.use_datamodule else None,
        learning_rate = args.learning_rate,       # Learning rate for optimizer
        input_channels = args.input_channels,     # Number of input channels
        data_path = args.data_path if not args.use_datamodule else None,
        impaired = args.impaired if not args.use_datamodule else None,
        qa = args.qa if not args.use_datamodule else None,
        datamodule = datamodule_param,            # Pass the datamodule
        checkpoint_path = args.checkpoint_path    # Path to checkpoint to resume training (if any)
    )

    # Train the model
    trainer.train()

if __name__ == '__main__':
    main()
