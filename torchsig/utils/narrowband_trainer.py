"""Narrowband Trainer for IQ Classification on Narrowband.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TorchSig imports
from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.transforms.transforms import (
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Compose,
)
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.datasets.datamodules import NarrowbandDataModule

# PyTorch Lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


class MetricsLogger(Callback):
    """
    PyTorch Lightning Callback to log training and validation metrics.

    Attributes:
        metrics (dict): A dictionary to store training and validation loss and accuracy.
    """

    def __init__(self):
        super().__init__()
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends. Logs the training loss and accuracy.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The model being trained.
        """
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.metrics['train_loss'].append(metrics['train_loss'].item())
        if 'train_acc' in metrics:
            self.metrics['train_acc'].append(metrics['train_acc'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called when the validation epoch ends. Logs the validation loss and accuracy.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (LightningModule): The model being validated.
        """
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics:
            self.metrics['val_loss'].append(metrics['val_loss'].item())
        if 'val_acc' in metrics:
            self.metrics['val_acc'].append(metrics['val_acc'].item())


class NarrowbandTrainer:
    """
    A trainer class for I/Q Signal Modulation Classification using narrowband datasets.

    This class encapsulates data preparation, model initialization, training,
    validation, prediction, and plotting functionalities.

    Attributes:
        model_name (str): Name of the model to use.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_workers (int): Number of workers for data loading.
        learning_rate (float): Learning rate for the optimizer.
        data_path (str): Path to the dataset.
        impaired (bool): Whether to use the impaired dataset.
        qa (bool): Whether to use QA configuration.
        checkpoint_path (str): Path to a checkpoint file to load the model weights.
        datamodule (LightningDataModule): Custom data module if provided.
        checkpoint_dir (str): Directory to save checkpoints.
        plots_dir (str): Directory to save plots.
        class_list (list): List of class names.
        num_classes (int): Number of classes.
        transform (Compose): Data transformations to apply.
        target_transform (DescToClassIndex): Target transformation.
        input_channels (int): Number of input channels.
        model (LightningModule): The model to train.
        trainer (Trainer): PyTorch Lightning trainer.
        metrics_logger (MetricsLogger): Callback for logging metrics.
        best_model_path (str): Path to the best saved model.
        filename_base (str): Base filename for saved plots.
    """

    def __init__(self, model_name='inception', num_epochs=10, batch_size=32,
                 num_workers=16, learning_rate=1e-3, input_channels = 2, data_path='../datasets/narrowband_test_QA',
                 impaired=True, qa=True, checkpoint_path=None, datamodule=None):
        """
        Initializes the NarrowbandTrainer with specified parameters.

        Args:
            model_name (str): Name of the model to use.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
            learning_rate (float): Learning rate for the optimizer.
            input_channels (int): Number of input channels into model.
            data_path (str): Path to the dataset.
            impaired (bool): Whether to use the impaired dataset.
            qa (bool): Whether to use QA configuration.
            checkpoint_path (str): Path to a checkpoint file to load the model weights.
            datamodule (LightningDataModule): Custom data module instance.
        """
        # Set random seed for reproducibility
        seed = 1234567890
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data Parameters
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.impaired = impaired
        self.qa = qa
        self.datamodule = datamodule  # Accept custom datamodule

        # Model Parameters
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path  # Added checkpoint_path

        # Other parameters
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.plots_dir = './plots'
        os.makedirs(self.plots_dir, exist_ok=True)

        # Prepare data module if not provided
        self.prepare_data()

        # Prepare model
        self.prepare_model()

        # Initialize trainer
        self.trainer = None  # Will be initialized in the train() method

    def prepare_data(self):
        """
        Prepares the data module for training and validation.

        Uses the provided datamodule or creates a new one if not provided.
        Sets up data transformations, target transformations, and initializes
        the NarrowbandDataModule.
        """
        if self.datamodule is None:
            # Get the class list and number of classes
            self.class_list = list(TorchSigNarrowband._idx_to_name_dict.values())
            self.num_classes = len(self.class_list)

            # Specify Transforms
            self.transform = Compose(
                [
                    RandomPhaseShift(phase_offset=(-1, 1)),
                    Normalize(norm=np.inf),
                    ComplexTo2D(),
                ]
            )
            self.target_transform = DescToClassIndex(class_list=self.class_list)

            self.datamodule = NarrowbandDataModule(
                root=self.data_path,
                qa=self.qa,
                impaired=self.impaired,
                transform=self.transform,
                target_transform=self.target_transform,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        else:
            # Use the provided datamodule
            self.class_list = self.datamodule.class_list
            self.num_classes = len(self.class_list)
            # Assume transforms are set within the provided datamodule
            print("Using custom datamodule provided.")

    def prepare_model(self):
        """
        Initializes the model based on the specified model name.

        If a checkpoint path is provided, loads the model directly from the checkpoint.

        Raises:
            ValueError: If the specified model name is not supported.
        """
        # Map model names to their corresponding classes
        self.available_models = {
            'xcit': 'XCiTClassifier',
            'inception': 'InceptionTime',
            'MyNewModel': 'MyNewModel',
        }

        if self.model_name not in self.available_models:
            raise ValueError(f"Model {self.model_name} is not supported.")

        # Determine the model class
        if self.model_name == 'xcit':
            from torchsig.models import XCiTClassifier
            ModelClass = XCiTClassifier
            model_kwargs = {
                'input_channels': self.input_channels,
                'num_classes': self.num_classes,
                'xcit_version': 'tiny_12_p16_224',
                'ds_method': 'downsample',
                'ds_rate': 16,
                'learning_rate': self.learning_rate,
            }
        elif self.model_name == 'inception':
            from torchsig.models import InceptionTime
            ModelClass = InceptionTime
            model_kwargs = {
                'input_channels': self.input_channels,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
            }
        elif self.model_name == 'MyNewModel':
            from my_models import MyNewModel
            ModelClass = MyNewModel
            model_kwargs = {
                'input_channels': self.input_channels,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
            }

        # Load model from checkpoint if provided
        if self.checkpoint_path:
            if not os.path.isfile(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file {self.checkpoint_path} not found.")
            # Load model directly from checkpoint
            self.model = ModelClass.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                **model_kwargs
            )
            print(f"Loaded model from checkpoint: {self.checkpoint_path}")
        else:
            # Initialize a new model instance
            self.model = ModelClass(**model_kwargs)

    def train(self):
        """
        Trains the model using the prepared data and model.

        If a checkpoint was loaded, continues training (fine-tuning) from the checkpoint.

        Sets up callbacks, initializes the PyTorch Lightning trainer, and
        starts the training process. After training, it plots metrics and
        the confusion matrix.
        """
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=self.checkpoint_dir,
            filename=self.model_name + '-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max',
        )

        # Metrics Logger Callback
        self.metrics_logger = MetricsLogger()

        # Trainer
        self.trainer = Trainer(
            max_epochs=self.num_epochs,
            callbacks=[checkpoint_callback, self.metrics_logger],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            # No need to specify resume_from_checkpoint when using load_from_checkpoint
        )

        # Train
        self.trainer.fit(self.model, self.datamodule)

        # Get the best checkpoint filename base
        self.best_model_path = checkpoint_callback.best_model_path
        self.filename_base = os.path.splitext(os.path.basename(self.best_model_path))[0]

        # Plot metrics
        self.plot_metrics()

        # Plot confusion matrix
        self.plot_confusion_matrix()

    def plot_metrics(self):
        """
        Plots training and validation loss and accuracy over epochs.

        Uses the metrics logged during training to create plots and saves
        them in the specified plots directory.
        """
        metrics = self.metrics_logger.metrics
        plots_dir = self.plots_dir
        filename_base = self.filename_base
        epochs = range(1, len(metrics['train_loss']) + 1)

        # Plot Loss
        plt.figure()
        plt.plot(epochs, metrics['train_loss'], label='Training Loss')

        if metrics['val_loss']:
            val_epochs = range(1, len(metrics['val_loss']) + 1)
            plt.plot(val_epochs, metrics['val_loss'], label='Validation Loss')
        else:
            print("Validation loss is empty. Skipping validation loss plot.")

        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        self.loss_plot_path = os.path.join(plots_dir, f'{filename_base}_loss.png')
        plt.savefig(self.loss_plot_path)
        plt.close()

        # Plot Accuracy
        plt.figure()
        plt.plot(epochs, metrics['train_acc'], label='Training Accuracy')

        if metrics['val_acc']:
            val_epochs = range(1, len(metrics['val_acc']) + 1)
            plt.plot(val_epochs, metrics['val_acc'], label='Validation Accuracy')
        else:
            print("Validation accuracy is empty. Skipping validation accuracy plot.")

        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        self.acc_plot_path = os.path.join(plots_dir, f'{filename_base}_accuracy.png')
        plt.savefig(self.acc_plot_path)
        plt.close()

    def plot_confusion_matrix(self):
        """
        Plots a normalized confusion matrix based on validation data.

        Saves the confusion matrix plot in the specified plots directory.
        """
        model = self.model
        datamodule = self.datamodule
        class_list = self.class_list
        plots_dir = self.plots_dir
        filename_base = self.filename_base

        # Ensure model is on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        val_loader = datamodule.val_dataloader()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.float().to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        if not all_preds or not all_labels:
            print("No predictions or labels available to plot confusion matrix.")
            return

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, normalize='true')

        # Set up the figure size to make labels readable
        plt.figure(figsize=(12, 10))

        # Plot the confusion matrix without annotations
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
        disp.plot(include_values=False, cmap='Blues', ax=plt.gca())

        # Increase font sizes for labels and ticks
        plt.title('Normalized Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)

        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()

        # Save the plot
        self.cm_plot_path = os.path.join(plots_dir, f'{filename_base}_confusion_matrix.png')
        plt.savefig(self.cm_plot_path, bbox_inches='tight', dpi=300)
        plt.close()

    def validate(self):
        """
        Validates the model using the validation dataset.

        This method can be customized to perform specific validation tasks
        and compute metrics as needed.
        """
        # Ensure model is on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        val_loader = self.datamodule.val_dataloader()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.float().to(device)
                y = y.to(device)
                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        accuracy = total_correct / total_samples
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        return accuracy

    def predict(self, data):
        """
        Predicts the class labels for the given data.

        Args:
            data (torch.Tensor): Input data tensor of shape [batch_size, channels, length].

        Returns:
            np.ndarray: Predicted class indices.
        """
        # Ensure model is on the correct device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            data = data.to(device)
            logits = self.model(data)
            preds = torch.argmax(logits, dim=1)
            return preds.cpu().numpy()
