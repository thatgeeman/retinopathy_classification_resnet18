import numpy as np
from matplotlib import pyplot as plt

from config import stats


def annotate_axes(ax, text, fontsize=10, x=0., y=1.1):
    """Add annotations to ax object."""
    ax.text(x, y, text, transform=ax.transAxes,
            va="center", fontsize=fontsize, color="gray")


def show_images(images, labels=None, preds=None, ncols=2, nrows=3, mean=stats["mean"], std=stats["std"]):
    """Show method to display images from the dataloader batch."""
    plt.figure(figsize=(8, 6))
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    mean = np.array(mean)
    std = np.array(std)
    for i, image in enumerate(images):
        plt.subplot(ncols, nrows, i + 1, xticks=[], yticks=[])
        image = image * std + mean
        if preds is not None and labels is not None:
            col = 'green' if preds[i] == labels[i] else 'red'
            true_label = f'{labels[i].detach().cpu().numpy()}'
            pred_label = f'{preds[i].detach().cpu().numpy()}'
            plt.xlabel(true_label)
            plt.ylabel(pred_label, color=col)
        plt.imshow(image.clip(0, 255))

    plt.tight_layout()
    plt.show()
