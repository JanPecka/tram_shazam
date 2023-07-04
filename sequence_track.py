import json
from argparse import ArgumentParser

import torch

from common.constants import SAVED_MODELS_PATH, SAVED_OUTPUTS_PATH
from common.parameters import DURATION_FOR_TRAINING
from data_structs.track import Track, TrackMetadata
from data_transformations.preprocessing_pipeline import PreprocessingPipeline
from inference.audio_sequencer import AudioSequencer
from models.spectrogram_model import SpectrogramModel

SEQUENCE_DURATION_S = DURATION_FOR_TRAINING
HOP_LENGTH_S = .1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=["spectre_1"], default="spectre_1")
    parser.add_argument("--track-path", type=str, help="Filepath of track to be sequenced.")
    args = parser.parse_args()
    print(f"Sequencing track {args.track_path} with model {args.model_name}.")

    model = SpectrogramModel()
    model.load_state_dict(torch.load(SAVED_MODELS_PATH.format(file_name=f"{args.model_name}.pth")))
    model.eval()
    with open(SAVED_MODELS_PATH.format(file_name=f"{args.model_name}_label_decoder.json"), "r") as fp:
        label_decoder = json.load(fp)

    preprocessing_pipeline = PreprocessingPipeline()
    track = Track.from_metadata(TrackMetadata(filepath=args.track_path))

    sequencer = AudioSequencer(model, preprocessing_pipeline, SEQUENCE_DURATION_S, HOP_LENGTH_S)
    predictions = sequencer.sequence(track)
    sequencer.plot_predictions(predictions, label_decoder)

    sequencer.save_output(SAVED_OUTPUTS_PATH.format(file_name=f"{args.model_name}.csv"), predictions, label_decoder)
