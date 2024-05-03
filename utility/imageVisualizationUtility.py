

# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail.png"



import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs, labels, fig_size=(10,4)):
    if not isinstance(imgs, list):
        imgs = [imgs]

    if not isinstance(labels, list):
        labels = [labels]

    if(len(labels) != len(imgs)):
        labels = ["-"] * len(imgs)

    figure = plt.figure(figsize=fig_size)

    rows = 2
    cols = math.ceil(len(imgs)/2)
   
    max_count = min(cols * rows, len(imgs))
    for i in range(1, max_count + 1):
        img = imgs[i-1]

        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = np.transpose(img.numpy(), (1, 2, 0))

        figure.add_subplot(rows, cols, i)  
        plt.axis("off")
        plt.title(labels[i-1])
        plt.imshow(np.asarray(img))

    plt.show()
