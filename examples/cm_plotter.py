import numpy as np
from typing import Optional
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    classes: list,
    normalize: bool = True,
    title: Optional[str] = None,
    text: bool = True,
    rotate_x_text: int = 90,
    figsize: tuple = (16,9),
    cmap: plt.cm = plt.cm.Blues,
):
    """Function to help plot confusion matrices
    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="none", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set( 
        xticks=np.arange(cm.shape[1]), 
        yticks=np.arange(cm.shape[0]), 
        xticklabels=classes, 
        yticklabels=classes, 
        title=title, 
        ylabel="True label", 
        xlabel="Predicted label",
    )
    ax.set_xticklabels(classes, rotation=rotate_x_text)
    ax.figure.set_size_inches(figsize)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if text:
                ax.text(j, i, format(cm[i,j], fmt), ha="center", va="center", color="white" if cm[i,j] > thresh else "black")
    if len(classes) == 2:
        plt.axis([-0.5, 1.5, 1.5, -0.5])
        fig.tight_layout()

    return ax
