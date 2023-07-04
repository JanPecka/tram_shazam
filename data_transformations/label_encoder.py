import numpy as np
from torch import Tensor, tensor

from data_structs.dataset import Dataset


class LabelEncoder:
    """A simple OHE of labels into numpy arrays.

    I'm sure there's a one-liner for this, but I've already spent more time looking for it than just writing it myself.
    """

    def __init__(self):
        self.labels: list[str] | None = None
        self.encodings: dict[str, np.ndarray] | None = None
        self.inverse_mapping: dict[int, str] | None = None

    def fit(self, dataset: Dataset) -> None:
        labels = sorted(dataset.unique_labels)
        self.labels = labels
        self.encodings = {l: np.insert(np.zeros(len(labels) - 1, np.float), i, 1) for i, l in enumerate(labels)}
        self.inverse_mapping = {i: l for i, l in enumerate(labels)}

    def transform(self, label: str) -> np.ndarray:
        if self.encodings is None:
            raise Exception("Please fit the label encoder first.")
        if label not in self.encodings:
            raise ValueError(f"Label {label} was not present during the encoders' fitting.")
        return self.encodings[label]

    def transform_torch_batch(self, batch_of_labels: list[str]) -> Tensor:
        return tensor(np.array([self.transform(l) for l in batch_of_labels]))

    def transform_i_into_label(self, i: int) -> str:
        return self.inverse_mapping[i]
