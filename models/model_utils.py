import json
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from common.constants import SAVED_MODELS_PATH
from data_structs.dataset import Dataset, train_test_val_split
from data_transformations.label_encoder import LabelEncoder


class ModelUtils:
    """
    Wrapper for commonly used PyTorch functionalities.
    """

    def __init__(self, model_name: str):
        self.name = model_name

    @staticmethod
    def train_loop(
        model: nn.Module,
        dataloader: DataLoader,
        label_encoder: LabelEncoder,
        loss_f: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One training epoch."""
        avg_loss = 0
        model.train()
        for X, y_labels in dataloader:
            pred = model(X)
            y_encoded = label_encoder.transform_torch_batch(y_labels)
            loss = loss_f(pred, y_encoded)
            avg_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss /= len(dataloader)
        print(f"Avg loss: {avg_loss:.3f}\n")
        return avg_loss

    @staticmethod
    def get_metrics(
        model: nn.Module,
        dataloader: DataLoader,
        label_encoder: LabelEncoder,
        loss_f: Callable,
        show_confusion_matrix: bool = False,
        set_type: str | None = None,
    ) -> float:
        """Calculate metrics on a dataset."""
        avg_loss = 0
        n_correct = 0
        targets = []
        predictions = []

        model.eval()
        with torch.no_grad():
            for X, y_labels in dataloader:
                pred = model(X)
                y_encoded = label_encoder.transform_torch_batch(y_labels)
                avg_loss += loss_f(pred, y_encoded).item()
                n_correct += (pred.argmax(1) == y_encoded.argmax(1)).type(torch.float).sum().item()
                if show_confusion_matrix:
                    targets += list(y_encoded.argmax(1))
                    predictions += list(pred.argmax(1))

        avg_loss /= len(dataloader)
        print(f"Avg loss: {avg_loss:.3f}, Accuracy: {100 * n_correct / len(dataloader.dataset):>0.1f}% \n")

        if show_confusion_matrix:
            c_m = confusion_matrix(targets, predictions)
            fig, ax = plt.subplots(1, 1, figsize=(16, 17))
            disp = ConfusionMatrixDisplay(c_m, display_labels=label_encoder.labels)
            disp.plot(xticks_rotation="vertical", ax=ax, colorbar=False)
            if set_type is not None:
                plt.title(f"{set_type} dataset")
            plt.show()

        return avg_loss

    @staticmethod
    def plot_losses(train: list[float], test: list[float]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(train, label="train loss")
        ax.plot(test, label="test loss")
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.show()

    def train(
        self,
        model: nn.Module,
        dataset: Dataset,
        label_encoder: LabelEncoder,
        loss_f: Callable,
        optimizer: torch.optim.Optimizer,
        train_set_fraction: float,
        test_set_fraction: float,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        random_seed: int,
        save_best: bool = False,
    ):
        """Train a model, calculate its metrics, plot loss on datasets during training, and save the best version."""
        train, test, val = train_test_val_split(dataset, train_set_fraction, test_set_fraction, random_seed)
        print(f"# train samples: {len(train)}, # test samples: {len(test)}, # val samples: {len(val)},")

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test, batch_size=len(test))
        val_dataloader = DataLoader(val, batch_size=len(val))

        optimizer.lr = learning_rate

        best_iter = 0
        best_loss = np.Inf
        best_model = None
        train_losses = []
        test_losses = []

        for i in range(1, n_epochs + 1):
            print(f"\n\nEpoch {i}")
            print("Train set metrics:")
            train_loss = self.train_loop(model, train_dataloader, label_encoder, loss_f, optimizer)
            print("Test set metrics:")
            test_loss = self.get_metrics(model, test_dataloader, label_encoder, loss_f)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss < best_loss:
                best_iter = i
                best_loss = test_loss
                best_model = deepcopy(model.state_dict())

        print("\nFinal metrics on train set:")
        self.get_metrics(model, train_dataloader, label_encoder, loss_f, show_confusion_matrix=True, set_type="train")
        print("Final metrics on test set:")
        self.get_metrics(model, test_dataloader, label_encoder, loss_f, show_confusion_matrix=True, set_type="test")

        self.plot_losses(train_losses, test_losses)

        print(f"Best test results achieved during epoch #{best_iter}, test loss = {best_loss:.6f}")
        if save_best:
            print(f"Saving best model as {SAVED_MODELS_PATH.format(file_name=f'{self.name}.pth')}")
            torch.save(best_model, SAVED_MODELS_PATH.format(file_name=f"{self.name}.pth"))
            with open(SAVED_MODELS_PATH.format(file_name=f"{self.name}_label_decoder.json"), "w") as fp:
                json.dump(label_encoder.inverse_mapping, fp)

        return val_dataloader
