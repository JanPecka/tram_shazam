import torch
import torch.nn as nn

from common.constants import DATASET_PATH, SAVED_MODELS_PATH
from common.parameters import DURATION_FOR_TRAINING, TEST_DATASET_FRACTION, TRAIN_DATASET_FRACTION
from data_structs.dataset import Dataset, DatasetMetadata
from data_transformations.label_encoder import LabelEncoder
from data_transformations.preprocessing_pipeline import PreprocessingPipeline
from models.model_utils import ModelUtils
from models.spectrogram_model import SpectrogramModel

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_EPOCHS = 65
REDUCE_NOISE = True
MODEL_NAME = "spectre_2_2"

if __name__ == "__main__":
    preprocessing_pipeline = PreprocessingPipeline()

    dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH)
    dataset = Dataset.load_all_from_metadata(
        dataset_metadata,
        preprocessing_pipeline,
        use_spectrograms=True,
        max_duration=DURATION_FOR_TRAINING,
        reduce_noise=REDUCE_NOISE,
    )

    model = SpectrogramModel()
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    label_encoder = LabelEncoder()
    label_encoder.fit(dataset)

    model_utils = ModelUtils(model_name=MODEL_NAME)
    val_dataloader = model_utils.train(
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
        random_seed=7,
        save_best=True,
    )

    # Test save/load and compute metrics on the holdout dataset.
    best_model = SpectrogramModel()
    best_model.load_state_dict(torch.load(SAVED_MODELS_PATH.format(file_name=f"{MODEL_NAME}.pth")))
    best_model.eval()
    print("\n\n------\nMetrics of saved model on validation dataset:")
    model_utils.get_metrics(best_model, val_dataloader, label_encoder, loss_f, True, set_type="val")

    # Spectre_1 results:
    # - lowest train set loss 0.045
    # - lowest test set loss 0.094, highest accuracy 97.5%
    # - results on validation set: loss 0.123, accuracy 96.1%

    # Spectre_2 results: (with noise reduction)
    # - lowest train set loss 0.073
    # - lowest test set loss 0.111, highest accuracy 96.5%
    # - results on validation set: loss 0.131, accuracy 96.5%
