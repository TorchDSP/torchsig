from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from torchsig.transforms.transforms import (
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Compose,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torchsig.utils.cm_plotter import plot_confusion_matrix
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import click
import torch
import os


class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader):
        super(ExampleNetwork, self).__init__()
        self.mdl: torch.nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.val_data_loader: DataLoader = val_data_loader

        # Hyperparameters
        self.lr = 0.001
        self.batch_size = data_loader.batch_size

    def forward(self, x: torch.Tensor):
        return self.mdl(x.float())

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.forward(x.float())
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def training_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


@click.command()
@click.option("--root", default="data/sig53", help="Path to train/val datasets")
@click.option("--impaired", default=False, help="Impaired or clean datasets")
def main(root: str, impaired: bool):
    class_list = list(Sig53._idx_to_name_dict.values())
    transform = Compose(
        [
            RandomPhaseShift(phase_offset=(-1, 1)),
            Normalize(norm=np.inf),
            ComplexTo2D(),
        ]
    )
    target_transform = DescToClassIndex(class_list=class_list)

    sig53_train = Sig53(
        root,
        train=True,
        impaired=impaired,
        transform=transform,
        target_transform=target_transform,
        use_signal_data=True,
    )

    sig53_val = Sig53(
        root,
        train=False,
        impaired=impaired,
        transform=transform,
        target_transform=target_transform,
        use_signal_data=True,
    )

    # Create dataloaders"data
    train_dataloader = DataLoader(
        dataset=sig53_train,
        batch_size=os.cpu_count(),
        num_workers=os.cpu_count() // 2,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=sig53_val,
        batch_size=os.cpu_count(),
        num_workers=os.cpu_count() // 2,
        shuffle=False,
        drop_last=True,
    )

    model = efficientnet_b4(pretrained=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    example_model = ExampleNetwork(model, train_dataloader, val_dataloader)
    example_model = example_model.to(device)

    # Setup checkpoint callbacks
    checkpoint_filename = "{}/checkpoint".format(os.getcwd())
    checkpoint_callback = ModelCheckpoint(
        filename=checkpoint_filename,
        save_top_k=True,
        monitor="val_loss",
        mode="min",
    )

    # Create and fit trainer
    epochs = 500
    trainer = Trainer(
        max_epochs=epochs, callbacks=checkpoint_callback, devices=1, accelerator="gpu"
    )
    trainer.fit(example_model)

    # Load best checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        checkpoint_filename + ".ckpt", map_location=lambda storage, loc: storage
    )
    example_model.load_state_dict(checkpoint["state_dict"])
    example_model = example_model.to(device=device).eval()

    # Infer results over validation set
    num_test_examples = len(sig53_val)
    num_classes = len(list(Sig53._idx_to_name_dict.values()))
    y_raw_preds = np.empty((num_test_examples, num_classes))
    y_preds = np.zeros((num_test_examples,))
    y_true = np.zeros((num_test_examples,))

    for i in tqdm(range(0, num_test_examples)):
        # Retrieve data
        idx = i  # Use index if evaluating over full dataset
        data, label = sig53_val[idx]
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
    plt.savefig("{}/02_sig53_classifier.png".format(os.getcwd()))

    print("Classification Report:")
    print(classification_report(y_true, y_preds))


if __name__ == "__main__":
    main()
