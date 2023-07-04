from random import sample, seed

import librosa
import matplotlib.pyplot as plt
import polars as pl

from common.constants import DATASET_PATH
from data_structs.dataset import Dataset, DatasetMetadata
from data_structs.track import Track


class DatasetExplorer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def analyse(self) -> None:
        """Prints out very basic characteristics of the dataset."""
        df = self.dataset.df
        print(f"\nThere are {df.shape[0]} tracks in the DF.")
        n_negative = df.filter(pl.col("is_negative")).shape[0]
        print(f"Out of these, {n_negative} ({n_negative / df.shape[0]:.1%}) are of negative noise.")
        df_trams = df.filter(~pl.col("is_negative"))
        print(f"\nFollowing statistics are for tracks containing tram sounds:")
        print(df_trams.groupby("movement_type", "tram_type").count().sort("movement_type", "tram_type"))
        print(df_trams.groupby("movement_type").count().sort("movement_type"))
        print(df_trams.groupby("tram_type").count().sort("tram_type"))

    def analyse_durations(self) -> None:
        df = self.dataset.df
        print(f"\nAverage track duration is {df.select('duration').mean().item(0, 0):.2f}s.")
        print(
            f"For negative noise, this avg is {df.filter(pl.col('is_negative')).select('duration').mean().item(0, 0):.2f}s."
        )
        df_trams = df.filter(~pl.col("is_negative"))
        print(f"For tram sounds, this avg is {df_trams.select('duration').mean().item(0, 0):.2f}s.")
        print(f"\nFollowing statistics are for tracks containing tram sounds:")
        print(
            df_trams.groupby("movement_type", "tram_type")
            .agg(pl.col("duration").mean())
            .sort("movement_type", "tram_type")
        )
        print(df_trams.groupby("movement_type").agg(pl.col("duration").mean()).sort("movement_type"))
        print(df_trams.groupby("tram_type").agg(pl.col("duration").mean()).sort("tram_type"))

    def histogram_of_durations(self) -> None:
        """Produces a histogram of track durations for each combination of tram type and movement type.

        The classes seem to be distributed fairly equally.
        """
        df_trams = self.dataset.df.filter(~pl.col("is_negative"))
        fig, ax = plt.subplots(1, 1)
        for tram, movement in df_trams.select("tram_type", "movement_type").unique().to_numpy():
            y = (
                df_trams.filter((pl.col("tram_type") == tram) & (pl.col("movement_type") == movement))
                .select("duration")
                .to_numpy()
            )
            plt.hist(y, density=False, histtype="step", label=f"{tram}_{movement}")
        plt.legend()
        plt.show()

    def duration_percentiles_per_class(self):
        """For each combination of tram and movement type, shows 95th percentile of duration."""
        # Polars do not have a simple function for percentiles by group, hence this workaround.
        df_duration_percentiles = (
            self.dataset.df.filter(~pl.col("is_negative"))
            .select(
                pl.concat_str("tram_type", "movement_type", separator="_").alias("class"),
                "duration",
            )
            .with_columns(
                pl.col("duration").rank("ordinal").over("class").alias("rank_in_class"),
                pl.col("class").count().over("class").alias("n_rows_in_class"),
            )
            .select("class", "duration", (pl.col("rank_in_class") / pl.col("n_rows_in_class")).alias("percentile"))
            .sort("class", "percentile")
        )

        print(
            df_duration_percentiles.filter(pl.col("percentile") > 0.95)
            .groupby("class")
            .agg(pl.col("duration").first().alias("duration_95_percentile"))
            .sort("class")
        )

    def display_grid_of_spectrograms(
        self,
        n_samples: int = 5,
        type: str = "mel",
        y_axis: str | None = "mel",
        n_fft: int = 2048,
        hop_length: int = 512,
        random_seed: int = 77,
        noise_track: Track | None = None,
    ) -> None:
        """Takes `n_sample` from each class and creates a grid of spectrograms."""
        seed(random_seed)
        if y_axis is None:
            y_axis = "mel" if type == "mel" else "linear"
        classes = {
            (t.metadata.tram_type, t.metadata.movement_type) for t in self.dataset.tracks if not t.metadata.is_negative
        }

        # Create a selection of tracks to display, `n_samples` for each class + `n_samples` negative tracks.
        # TODO: This thing first loads all the tracks and only then does it create a sample, should be done oppositely.
        track_selection: list[Track] = sample(
            [t for t in self.dataset.tracks if t.metadata.is_negative],
            n_samples,
        )
        for t_t, m_t in sorted(classes, key=lambda t: (t[1], t[0])):
            track_selection.extend(
                sample(
                    [
                        t
                        for t in self.dataset.tracks
                        if not t.metadata.is_negative
                        and t.metadata.tram_type == t_t
                        and t.metadata.movement_type == m_t
                    ],
                    n_samples,
                )
            )

        fig, axs = plt.subplots(n_samples, len(classes) + 1, squeeze=False, figsize=(22, 12.5), sharey="all")
        for ax, t in zip(axs.transpose().flatten(), track_selection):
            if noise_track is not None:
                t.reduce_noise(noise_track)
            spectrogram = t.create_spectrogram(type=type, n_fft=n_fft, hop_length=hop_length)
            img = librosa.display.specshow(spectrogram, y_axis=y_axis, ax=ax)
            if t.metadata.is_negative:
                title = "negative"
            else:
                title = f"{t.metadata.tram_type} {t.metadata.movement_type}"
            ax.set(title=title)
        plt.show()

    def run(self) -> None:
        self.analyse()
        self.analyse_durations()
        self.histogram_of_durations()
        self.duration_percentiles_per_class()
        self.display_grid_of_spectrograms()


if __name__ == "__main__":
    random_seed = 77

    dataset_metadata = DatasetMetadata.from_directory(DATASET_PATH)
    dataset = Dataset.load_all_from_metadata(dataset_metadata)
    explorer = DatasetExplorer(dataset)
    explorer.run()
