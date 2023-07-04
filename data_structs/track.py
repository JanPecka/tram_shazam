from dataclasses import asdict, dataclass
from random import randint, seed
from typing import Self

import librosa
import matplotlib.pyplot as plt
import numpy as np
from noisereduce import reduce_noise

from common.constants import DATASET_PATH
from common.parameters import INPUT_SAMPLING_RATE


@dataclass
class TrackMetadata:
    """
    Class containing available information about a track.
    """
    filepath: str
    tram_type: str | None = None
    movement_type: str | None = None
    is_negative: bool | None = None
    sampling_rate: int = INPUT_SAMPLING_RATE

    @property
    def label(self) -> str:
        return f"{self.tram_type}_{self.movement_type}" if not self.is_negative else "negative"


@dataclass
class Track:
    """
    Reprentation of an audio file.
    """
    metadata: TrackMetadata
    signal: np.ndarray
    _spectrogram: np.ndarray | None = None
    duration: float | None = None

    @classmethod
    def from_metadata(cls, metadata: TrackMetadata) -> Self:
        signal, _ = librosa.load(metadata.filepath, sr=metadata.sampling_rate)
        duration = librosa.get_duration(y=signal, sr=metadata.sampling_rate)
        return cls(metadata=metadata, signal=signal, duration=duration)

    def __len__(self) -> int:
        return len(self.signal)

    def to_dict(self) -> dict:
        return {
            **asdict(self.metadata),
            **{"signal": self.signal, "duration": self.duration},
        }

    def pad_to_duration(self, target_duration: float) -> None:
        if self.duration > target_duration:
            return
        size = int(target_duration * self.metadata.sampling_rate)
        self.signal = librosa.util.pad_center(self.signal, size=size)
        self.duration = target_duration

    def trim(self, top_db: int = 60) -> None:
        signal_trimmed, _ = librosa.effects.trim(self.signal, top_db=top_db)
        self.signal = signal_trimmed
        self.duration = librosa.get_duration(y=signal_trimmed, sr=self.metadata.sampling_rate)

    def reduce_noise(self, noise_track: Self) -> None:
        signal_wo_noise = reduce_noise(
            self.signal, sr=self.metadata.sampling_rate, y_noise=noise_track.signal, n_fft=2048
        )
        self.signal = signal_wo_noise

    def random_subset(self, target_duration: float, random_seed: int | None = None) -> None:
        """Randomly trim the track to be `target_duration` seconds long."""
        if self.duration < target_duration:
            return
        if random_seed is not None:
            seed(random_seed)
        n_samples = int(target_duration * self.metadata.sampling_rate)
        subset_start = randint(0, len(self.signal) - n_samples)
        self.signal = self.signal[subset_start : (subset_start + n_samples)]
        self.duration = librosa.get_duration(y=self.signal, sr=self.metadata.sampling_rate)

    def display_waveform(self) -> None:
        fig, ax = plt.subplots(1, 1)
        librosa.display.waveshow(self.signal, sr=self.metadata.sampling_rate, ax=ax)
        plt.suptitle(f"{self.metadata.tram_type} {self.metadata.movement_type}")
        plt.title(self.metadata.filepath)
        plt.show()

    def create_spectrogram(
        self,
        type: str = "mel",
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        if type == "mel":
            s = librosa.feature.melspectrogram(
                y=self.signal,
                sr=self.metadata.sampling_rate,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            s = librosa.power_to_db(s, ref=np.max)
        elif type == "stft":
            s = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)
            s = librosa.amplitude_to_db(np.abs(s), ref=np.max)
        else:
            raise ValueError("Please provide a supported type of spectrogram.")
        return s

    def display_spectrogram(
        self,
        s: np.ndarray | None = None,
        type: str = "mel",
        y_axis: str | None = None,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        if s is None:
            s = self.create_spectrogram(type, n_fft, hop_length)
        if y_axis is None:
            y_axis = "mel" if type == "mel" else "linear"
        fig, ax = plt.subplots(1, 1)
        img = librosa.display.specshow(s, x_axis="time", y_axis=y_axis, ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        if self.metadata.is_negative:
            title = "negative"
        else:
            title = f"{self.metadata.tram_type} {self.metadata.movement_type}"
        ax.set(title=title)
        plt.show()


def get_track_subset(track: Track, i_start: int, i_end: int) -> Track:
    """Create a track that is a subset of parent track between indexes `i_start` and `i_end."""
    signal = track.signal[i_start:i_end]
    duration = librosa.get_duration(y=signal)
    return Track(metadata=track.metadata, signal=signal, duration=duration)


if __name__ == "__main__":
    from random import choice

    from data_structs.dataset import DatasetMetadata

    seed(7777)

    tram_dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH, track_type="tram")
    noise_dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH, track_type="negative")

    track_metadata = choice(tram_dataset_metadata.tracks_metadata)
    track = Track.from_metadata(track_metadata)

    noise_track_metadata = choice(noise_dataset_metadata.tracks_metadata)
    noise_track = Track.from_metadata(noise_track_metadata)

    track.display_waveform()
    track.display_spectrogram()
    track.reduce_noise(noise_track)
    track.display_waveform()
    track.display_spectrogram()
