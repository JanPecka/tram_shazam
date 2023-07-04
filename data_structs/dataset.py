from dataclasses import asdict, dataclass
from pathlib import Path
from random import choice, seed
from typing import Self

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from common.constants import DATASET_PATH
from common.parameters import TEST_DATASET_FRACTION, TRAIN_DATASET_FRACTION
from data_structs.track import Track, TrackMetadata
from data_transformations.preprocessing_pipeline import PreprocessingPipeline

TRAM_TYPE_MAP = {
    "1_New": "new",
    "2_CKD_Long": "ckd_long",
    "3_CKD_Short": "ckd_short",
    "4_Old": "old",
    "checked": None,
}
MOVEMENT_TYPE_MAP = {
    "accelerating": "accelerating",
    "braking": "braking",
    "negative": None,
}


@dataclass
class DatasetMetadata:
    """
    Helper class for parsing the input dataset.
    """
    tracks_metadata: list[TrackMetadata]

    @classmethod
    def from_directory(cls, root_dir: str, track_type: str = None) -> Self:
        """Walks through the directory and loads metadata for all present tracks.

        When `track_type` is equal to `tram`, only tram tracks are loaded, when it is equal to `negative`, only negative noise is loaded.
        """
        if track_type is not None and track_type not in ["tram", "negative"]:
            raise ValueError(f"Argument track_type must be one of {', '.join(['tram', 'negative'])}")
        path = Path(root_dir)
        audio_file_paths = path.rglob("*.wav")
        metadata = [
            TrackMetadata(
                tram_type=TRAM_TYPE_MAP[p.parts[3]],
                movement_type=MOVEMENT_TYPE_MAP[p.parts[2]],
                filepath=p.as_posix(),
                is_negative=True if p.parts[2] == "negative" else False,
            )
            for p in audio_file_paths
            if (track_type is None)
            or (track_type == "negative" and p.parts[2] == "negative")
            or (track_type == "tram")
            and p.parts[2] in ["accelerating", "braking"]
        ]
        return cls(tracks_metadata=metadata)

    def to_polars(self) -> pl.DataFrame:
        df = pl.DataFrame(asdict(t) for t in self.tracks_metadata)
        return df


@dataclass
class Dataset(TorchDataset):
    """
    A PyTorch dataset containing a set of tracks and their characteristic.
    """
    tracks: list[Track] | None = None
    preprocessing_pipeline: PreprocessingPipeline | None = None
    _df: pl.DataFrame | None = None
    use_spectrograms: bool = False  # If True, returns spectrograms for training. Otherwise, returns plain signal.

    def __init__(
        self,
        tracks: list[Track] | None = None,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
        df: pl.DataFrame | None = None,
        use_spectrograms: bool = False,
        reduce_noise: bool = False,
    ):
        self.tracks = tracks
        self.preprocessing_pipeline = preprocessing_pipeline
        self.use_spectrograms = use_spectrograms
        self.reduce_noise = reduce_noise
        self._df = df

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx: int) -> (torch.Tensor, str):
        track = self.tracks[idx]
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline(track, noise_track=load_random_track("negative") if self.reduce_noise else None)

        if not self.use_spectrograms:
            x = torch.from_numpy(track.signal)
        else:
            x = torch.from_numpy(track.create_spectrogram()[np.newaxis, :])  # New dimension to simulate mono channel.
        y = track.metadata.label

        return x, y

    @classmethod
    def load_all_from_metadata(
        cls,
        metadata: DatasetMetadata,
        preprocessing_pipeline: PreprocessingPipeline | None = None,
        use_spectrograms: bool = False,
        reduce_noise: bool = False,
        max_duration: float = None,
    ) -> Self:
        """Loads all tracks into memory."""
        tracks = [Track.from_metadata(tm) for tm in metadata.tracks_metadata]
        if max_duration is not None:  # First load, then throw away. Not ideal, but ok for our small dataset.
            tracks = [t for t in tracks if t.metadata.is_negative or t.duration <= max_duration]
        return cls(
            tracks=tracks,
            preprocessing_pipeline=preprocessing_pipeline,
            use_spectrograms=use_spectrograms,
            reduce_noise=reduce_noise,
        )

    def to_polars(self) -> pl.DataFrame:
        df = pl.DataFrame(t.to_dict() for t in self.tracks)
        return df

    @property
    def df(self) -> pl.DataFrame:
        if self._df is None:
            df = self.to_polars()
            self._df = df
        return self._df

    @property
    def unique_labels(self) -> list[str]:
        return list(set(t.metadata.label for t in self.tracks))

    def get_signals(self) -> list[np.ndarray]:
        return [t.signal for t in self.tracks]

    def get_labels(self) -> list[str]:
        return [t.metadata.label for t in self.tracks]


def load_random_track(track_type: str | None = None, random_seed: int | None = None) -> Track:
    if random_seed is not None:
        seed(random_seed)
    dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH, track_type=track_type)
    track_metadata = choice(dataset_metadata.tracks_metadata)
    return Track.from_metadata(track_metadata)


def train_test_val_split(
    dataset: Dataset, train_size: float, test_size: float, random_seed: int | None = None
) -> (Dataset, Dataset, Dataset):
    """Returns training, test, and validation datasets."""
    if train_size + test_size >= 1:
        raise ValueError("train_size + test_size has to be smaller than 1.0.")
    tracks_train, tracks_test = train_test_split(
        dataset.tracks,
        train_size=train_size,
        random_state=random_seed,
    )
    tracks_test, tracks_val = train_test_split(
        tracks_test,
        train_size=test_size / (1 - train_size),
        random_state=random_seed,
    )
    return (  # Resulting datasets inherit characteristics of input dataset.
        Dataset(
            tracks_train,
            dataset.preprocessing_pipeline,
            use_spectrograms=dataset.use_spectrograms,
            reduce_noise=dataset.reduce_noise,
        ),
        Dataset(
            tracks_test,
            dataset.preprocessing_pipeline,
            use_spectrograms=dataset.use_spectrograms,
            reduce_noise=dataset.reduce_noise,
        ),
        Dataset(
            tracks_val,
            dataset.preprocessing_pipeline,
            use_spectrograms=dataset.use_spectrograms,
            reduce_noise=dataset.reduce_noise,
        ),
    )
