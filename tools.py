import numpy as np
import torch
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas


def set_seed(seed=10):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return random_state


class ConfusionMatrix:
    def __init__(self):
        self.Data = []

    def apdate(self, outputs, labels):
        if not len(self.Data):
            self.Data = torch.zeros((outputs.shape[-1], outputs.shape[-1]))
        label = labels.to(torch.long)
        out = torch.argmax(outputs, axis=1)
        for i, j in zip(out, label):
            self.Data[i][j] = self.Data[i][j] + 1

    def get(self):
        return self.Data


def top_k_corrects(outputs, labels, k: int = 1):
    """
    outputs : (batchsise, n_class)
    labels : (batchsise,)
    """
    labels_dim = 1
    assert 1 <= k
    if k >= outputs.size(labels_dim):
        return outputs.size(0)
    k_labels = torch.topk(
        input=outputs, k=k, dim=labels_dim, largest=True, sorted=True)[1]
    running_corrects = 0
    for i in range(labels.size(0)):
        running_corrects += float(labels[i] in k_labels[i])
    return running_corrects


def creat_snp_folder_path(
    snp_path_0="snp/", model_name="model_name", Dataset_name="Dataset_name"
):
    snp_path = snp_path_0 + Dataset_name + "/" + model_name + "/"
    os.makedirs(snp_path, exist_ok=True)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y")
    tm_string = now.strftime("/%H_%M_%S/")
    snp_path = snp_path + dt_string + tm_string
    os.makedirs(snp_path)
    return snp_path


def deleat_old_model(snp_path, mask):
    for path, _, filenames in os.walk(snp_path):
        for file in filenames:
            if mask in file:
                os.remove(os.path.join(path, file))


def show_train(snp):
    Data = pandas.read_csv(snp)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = list(Data["Epoch"])
    t = []
    for loss in Data.columns[1::2]:
        y = list(Data[loss])
        ax1.plot(x, y)
    ax1.grid(True, color="green")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y", which="major", labelcolor="green")
    ax1.set_title("Динамика Loss")
    plt.legend(Data.columns[1::2], loc="upper right")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = list(Data["Epoch"])
    t = []
    for loss in Data.columns[2::2]:
        y = list(Data[loss])
        ax1.plot(x, y)
    ax1.grid(True, color="green")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Accuracy")
    ax1.tick_params(axis="y", which="major", labelcolor="green")
    ax1.set_title("Динамика Accuracy")
    plt.legend(Data.columns[2::2], loc="upper right")
    plt.show()

            
def freezing_body_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            print(name, layer)  
            names = name.split(sep='.')
            atr=model
            for i in names:
                atr=getattr(atr, i)
    atr.weight.requires_grad = True
    atr.bias.requires_grad = True
    return model