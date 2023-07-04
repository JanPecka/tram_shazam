import torch.nn as nn

from common.parameters import DURATION_FOR_TRAINING, INPUT_SAMPLING_RATE
from data_structs.dataset import Dataset


class BaselineModel(nn.Module):
    def __init__(
        self,
        data: Dataset,
    ):
        super(BaselineModel, self).__init__()

        signal_length = DURATION_FOR_TRAINING * INPUT_SAMPLING_RATE
        self.input_dataset = data
        self.model = nn.Sequential(
            nn.Linear(signal_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, len(data.unique_labels)),
        )

    def forward(self, x):
        out = self.model(x)
        return out
