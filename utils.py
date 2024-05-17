import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from tqdm import tqdm
from torch import Tensor
from numpy import ndarray
from typing import Tuple
import itertools
import matplotlib.pyplot as plt
import numpy as np

plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import warnings

warnings.filterwarnings("ignore")

# import xgboost as xgb
# import sklearn
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CM_WEIGHTS = torch.tensor(
    [[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32, requires_grad=False
)


def cm_loss(logits: Tensor, targets: Tensor, alpha: float = 1.0, beta: float = 1.0):
    weights = CM_WEIGHTS[targets]
    assert (
        alpha >= 0 and beta >= 0 and alpha + beta > 0
    ), "The alpha and beta parameters must be non-negative and at least one of them must be positive."

    return (
        alpha * F.cross_entropy(logits, targets)
        + beta * (1 - F.softmax(logits, dim=1).log() * weights).sum(dim=1).mean()
    )


class CustomCMLoss:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, targets: ndarray, logits: ndarray):
        logits = torch.tensor(logits, requires_grad=True)
        targets = torch.tensor(targets, dtype=int, requires_grad=False)
        loss = cm_loss(logits, targets, self.alpha, self.beta)
        loss.backward(retain_graph=True)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        # print(f"The shape of the gradient is {grad.shape}")

        # def func(x):
        #     return cm_loss(x, targets, self.alpha, self.beta)

        # hess = torch.autograd.functional.hessian(func, logits).diag().detach().numpy()
        # print(
        #     f"The hess is {torch.autograd.grad(grad.sum(), logits, create_graph=True)}"
        # )
        hess = torch.autograd.grad(grad.sum(), logits, create_graph=True)[0]

        grad = grad.detach().numpy().reshape(-1, 1)
        hess = hess.detach().numpy().reshape(-1, 1)
        # print(grad.shape, hess.shape)
        return grad, hess


def get_confusion_matrix(logits: Tensor, targets: Tensor):
    preds = logits.argmax(dim=1)
    cm = torch.zeros(3, 3)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    return cm


def get_confusion_matrix_xgb(preds: ndarray, targets: ndarray):
    cm = np.zeros((3, 3))
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    return cm


def evaluate(logits: Tensor, targets: Tensor):
    cm = get_confusion_matrix(logits, targets)
    metric = ((cm * CM_WEIGHTS) / targets.size(0)).sum().item()

    return metric


def evaluate_xgb(preds: ndarray, targets: ndarray):
    cm = get_confusion_matrix_xgb(preds, targets)
    metric = ((cm * CM_WEIGHTS.numpy()) / targets.shape[0]).sum()

    return metric


def train_with_performance(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    val_dataset: Tuple[ndarray] = None,
    eval_every: int = 10,
) -> float:
    model.train()
    running_loss = 0.0
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        for i, data in enumerate(progress_bar):
            inputs, labels, perf = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
            )
            optimizer.zero_grad()
            logits, perf_hat = model(inputs)
            loss = criterion(logits, labels, alpha, beta) + gamma * F.mse_loss(
                perf_hat, perf
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(
                loss=running_loss / (i + 1), metric=evaluate(logits, labels)
            )

        if (epoch + 1) % eval_every == 0:
            if val_dataset is not None:
                inputs, targets, _ = val_dataset
                # val_p_bar = tqdm(range(1), desc="Validation", leave=True)
                logits = model(inputs)[0]
                val_metric = evaluate(logits, targets)
                print(f"Epoch {epoch}: Validation metric: {val_metric:.4f}")
                # val_p_bar.set_postfix(val_metric=val_metric)


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=True
):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    plt.show()
