# # Example 02 - Sig53 Classifier
# This notebook walks through a simple example of how to use the clean Sig53 dataset, load a pre-trained supported model, and evaluate the trained network's performance. Note that the experiment and the results herein are not to be interpreted with any significant value but rather serve simply as a practical example of how the `torchsig` dataset and tools can be used and integrated within a typical [PyTorch](https://pytorch.org/) and/or [PyTorch Lightning](https://www.pytorchlightning.ai/) workflow.

# ----
# ### Import Libraries
# First, import all the necessary public libraries as well as a few classes from the `torchsig` toolkit. An additional import from the `cm_plotter.py` helper script is also done here to retrieve a function to streamline plotting of confusion matrices.

from torchsig.transforms.target_transforms.target_transforms import DescToClassIndex
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from torchsig.utils.writer import DatasetLoader, DatasetCreator, LMDBDatasetWriter
from torchsig.transforms.wireless_channel.wce import RandomPhaseShift
from torchsig.transforms.signal_processing.sp import Normalize
from torchsig.transforms.expert_feature.eft import ComplexTo2D
from torchsig.transforms.transforms import Compose
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from cm_plotter import plot_confusion_matrix
from torchsig.datasets.sig53 import Sig53
from torchsig.datasets.modulations import ModulationsDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchsig.datasets import conf
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import os


# ----
# ### Instantiate Sig53 Dataset
# Here, we instantiate the Sig53 clean training dataset and the Sig53 clean validation dataset. We demonstrate how to compose multiple TorchSig transforms together, using a data impairment with a random phase shift that uniformly samples a phase offset between -1 pi and +1 pi. The next transform normalizes the complex tensor, and the final transform converts the complex data to a real-valued tensor with the real and imaginary parts as two channels. We additionally provide a target transform that maps the `SignalDescription` objects, that are part of `SignalData` objects, to a desired format for the model we will train. In this case, we use the `DescToClassIndex` target transform to map class names to their indices within an ordered class list. Finally, we sample from our datasets and print details in order to confirm functionality.
#
# For more details on the Sig53 dataset instantiations, please see the Sig53 example notebook.

# Specify Sig53 Options
root = "sig53"
train = False
impaired = False
class_list = list(Sig53._idx_to_name_dict.values())
transform = Compose(
    [
        RandomPhaseShift(phase_offset=(-1, 1)),
        Normalize(norm=np.inf),
        ComplexTo2D(),
    ]
)
target_transform = DescToClassIndex(class_list=class_list)

# Instantiate the Sig53 Clean Training Dataset
cfg = conf.Sig53CleanTrainQAConfig

ds = ModulationsDataset(
    level=cfg.level,
    num_samples=cfg.num_samples,
    num_iq_samples=cfg.num_iq_samples,
    use_class_idx=cfg.use_class_idx,
    include_snr=cfg.include_snr,
    eb_no=cfg.eb_no,
)

loader = DatasetLoader(ds, seed=12345678)
writer = LMDBDatasetWriter(path="examples/sig53/sig53_clean_train")
creator = DatasetCreator(loader, writer)
creator.create()
sig53_clean_train = Sig53(
    "examples/sig53",
    train=True,
    impaired=False,
    transform=transform,
    target_transform=target_transform,
    use_signal_data=True,
)

# Instantiate the Sig53 Clean Validation Dataset
cfg = conf.Sig53CleanValQAConfig

ds = ModulationsDataset(
    level=cfg.level,
    num_samples=cfg.num_samples,
    num_iq_samples=cfg.num_iq_samples,
    use_class_idx=cfg.use_class_idx,
    include_snr=cfg.include_snr,
    eb_no=cfg.eb_no,
)

loader = DatasetLoader(ds, seed=12345678)
writer = LMDBDatasetWriter(path="examples/sig53/sig53_clean_val")
creator = DatasetCreator(loader, writer)
creator.create()
sig53_clean_val = Sig53(
    "examples/sig53",
    train=True,
    impaired=False,
    use_signal_data=True,
    transform=transform,
    target_transform=target_transform,
)

# Retrieve a sample and print out information to verify
idx = np.random.randint(len(sig53_clean_train))
data, label = sig53_clean_train[idx]
print("Dataset length: {}".format(len(sig53_clean_train)))
print("Data shape: {}".format(data.shape))
print("Label Index: {}".format(label))
print("Label Class: {}".format(Sig53.convert_idx_to_name(label)))


# ----
# ### Format Dataset for Training
# Next, the datasets are then wrapped as `DataLoaders` to prepare for training.

# Create dataloaders
train_dataloader = DataLoader(
    dataset=sig53_clean_train,
    batch_size=16,
    num_workers=8,
    shuffle=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=sig53_clean_val,
    batch_size=16,
    num_workers=8,
    shuffle=False,
    drop_last=True,
)


# ----
# ### Instantiate Supported TorchSig Model
# Below, we load a pretrained EfficientNet-B4 model, and then conform it to a PyTorch LightningModule for training.
pretrained = False if not os.path.exists("examples/efficientnet_b4.pt") else True

model = efficientnet_b4(
    pretrained=pretrained,
    path="examples/efficientnet_b4.pt",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader):
        super(ExampleNetwork, self).__init__()
        self.mdl = model
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = data_loader.batch_size

    def forward(self, x):
        return self.mdl(x.float())

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x.float())
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.data_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("loss", loss)

    def val_dataloader(self):
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self) -> None:
        # val_loss_mean = sum([o["val_loss"] for o in outputs]) / len(outputs)
        # self.log("val_loss", val_loss_mean, prog_bar=True)
        return super().on_validation_epoch_end()


example_model = ExampleNetwork(model, train_dataloader, val_dataloader)


# ----
# ### Train the Model
# To train the model, we first create a `ModelCheckpoint` to monitor the validation loss over time and save the best model as we go. The network is then instantiated and passed into a `Trainer` to kick off training.

# Setup checkpoint callbacks
checkpoint_filename = "{}/examples/checkpoints/checkpoint".format(os.getcwd())
checkpoint_callback = ModelCheckpoint(
    filename=checkpoint_filename,
    save_top_k=True,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

# Create and fit trainer
epochs = 25
trainer = Trainer(
    max_epochs=epochs, callbacks=checkpoint_callback, devices=1, accelerator="gpu"
)
trainer.fit(example_model)


# ----
# ### Evaluate the Trained Model
# After the model is trained, the checkpoint's weights are loaded into the model and the model is put into evaluation mode. The validation set is looped through, inferring results for each example and saving the predictions and the labels. Finally, the labels and predictions are passed into our confusion matrix plotting function to view the results and also passed into the `sklearn.metrics.classification_report` method to print metrics of interest.

# Load best checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    checkpoint_filename + ".ckpt", map_location=lambda storage, loc: storage
)
example_model.load_state_dict(checkpoint["state_dict"])
example_model = example_model.to(device=device).eval()

# Infer results over validation set
num_test_examples = len(sig53_clean_val)
num_classes = len(list(Sig53._idx_to_name_dict.values()))
y_raw_preds = np.empty((num_test_examples, num_classes))
y_preds = np.zeros((num_test_examples,))
y_true = np.zeros((num_test_examples,))

for i in tqdm(range(0, num_test_examples)):
    # Retrieve data
    idx = i  # Use index if evaluating over full dataset
    data, label = sig53_clean_val[idx]
    # Infer
    data = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
    pred_tmp = example_model.predict(data)
    pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
    # Argmax
    y_preds[i] = np.argmax(pred_tmp)
    # Store label
    y_true[i] = label


acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
plot_confusion_matrix(
    y_true,
    y_preds,
    classes=class_list,
    normalize=True,
    title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
        acc * 100
    ),
    text=False,
    rotate_x_text=90,
    figsize=(16, 9),
)
plt.savefig("examples/figures/02_sig53_classifier.png")

print("Classification Report:")
print(classification_report(y_true, y_preds))
