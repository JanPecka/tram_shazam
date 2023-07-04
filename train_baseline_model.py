import torch
import torch.nn as nn

from common.constants import DATASET_PATH
from common.parameters import TEST_DATASET_FRACTION, TRAIN_DATASET_FRACTION
from data_structs.dataset import Dataset, DatasetMetadata
from data_transformations.label_encoder import LabelEncoder
from data_transformations.preprocessing_pipeline import PreprocessingPipeline
from models.baseline_model import BaselineModel
from models.model_utils import ModelUtils

BATCH_SIZE = 200
LEARNING_RATE = 1e-2
N_EPOCHS = 2

if __name__ == "__main__":
    preprocessing_pipeline = PreprocessingPipeline()

    dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH)
    dataset = Dataset.load_all_from_metadata(
        dataset_metadata, preprocessing_pipeline=preprocessing_pipeline, use_spectrograms=False
    )

    model = BaselineModel(dataset)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset)

    model_utils = ModelUtils(model_name="baseline")
    model_utils.train(
        model,
        dataset,
        label_encoder,
        loss_f,
        optimizer,
        TRAIN_DATASET_FRACTION,
        TEST_DATASET_FRACTION,
        BATCH_SIZE,
        N_EPOCHS,
        LEARNING_RATE,
        random_seed=77,
    )
