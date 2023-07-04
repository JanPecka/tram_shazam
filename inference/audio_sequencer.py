import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from data_structs.track import Track, get_track_subset
from data_transformations.preprocessing_pipeline import PreprocessingPipeline


class AudioSequencer:
    def __init__(
        self,
        model: nn.Module,
        preprocessing_pipeline: PreprocessingPipeline,
        sequence_duration_s: float,
        stride_length_s: float,
    ):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.seq_duration = sequence_duration_s
        self.stride_length = stride_length_s

    def sequence(self, track: Track) -> np.ndarray:
        """Sequence the given track. For each window, return class probabilities for each class."""
        n_samples_in_seq = int(self.seq_duration * track.metadata.sampling_rate)
        n_samples_in_stride = int(self.stride_length * track.metadata.sampling_rate)

        i = 0
        res = []
        seq_start_i = 0
        print("Creating sequence predictions.")
        while seq_start_i + n_samples_in_seq <= len(
            track
        ):  # End of track not reached. TODO: slide window across beginning and end?
            seq = get_track_subset(track, seq_start_i, seq_start_i + n_samples_in_seq)
            self.preprocessing_pipeline(seq)
            seq_spectrogram = seq.create_spectrogram()

            pred = self.model(torch.from_numpy(seq_spectrogram)[np.newaxis, np.newaxis, :])
            pred_prob = nn.Softmax(1)(pred)
            res.append(pred_prob.detach().numpy()[0])

            i += 1
            seq_start_i += n_samples_in_stride
            if i % 100 == 0:
                print(f"progress: {i / (len(track) / n_samples_in_stride):.1%}")

        print("Sequencing finished.")
        return np.array(res)

    def plot_predictions(self, prob: np.ndarray, label_decoder: dict[str]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(32, 18))
        x_ax = [x * self.stride_length for x in range(prob.shape[0])]
        for i in range(prob.shape[1]):
            ax.plot(x_ax, prob[:, i], label=label_decoder[str(i)])
        plt.xlabel("time [s]")
        plt.xticks(np.arange(0, max(x_ax) + 1, 10))
        plt.ylabel("prediction")
        plt.legend()
        plt.show()

    def save_output(
        self,
        path: str,
        prob: np.ndarray,
        label_decode: dict[str],
        min_tram_prob: float = 0.3,  # Higher probabilities are flagged as tram detection.
        max_negative_prob: float = 0.85,  # Lower probabilities are flagged as not negative.
    ) -> None:
        """Parse the predictions into the desired output format.

        `prob` should be an array of size (n_sequences, n_classes)
        """
        df = pl.DataFrame(prob, schema=[label_decode[str(i)] for i in range(prob.shape[1])])

        df_rolling_avg = df.select(  # Smooth out neighbouring predictions.
            pl.col(c).rolling_mean(window_size=5, center=True).alias(c) for c in df.columns
        )

        df_tram_detection = (
            df_rolling_avg.select(
                ((pl.col(c) > min_tram_prob).cast(bool).alias(f"is_{c}") for c in df.columns if c != "negative"),
                # (pl.col("negative") < max_negative_prob).cast(bool).alias("not_negative"),
            )
            .with_columns(pl.Series(values=[x * self.stride_length for x in range(prob.shape[0])], name="time"))
            .drop_nulls()  # First and last row, as a result of the rolling average.
        )

        df_out = df_tram_detection.filter(pl.any(c for c in df_tram_detection.columns if c != "time")).select(
            pl.col("time").cast(int).cast(str),  # First round, then cast to string.
            *[pl.col(c).cast(int).alias(c.lstrip("is_")) for c in df_tram_detection.columns if c != "time"],
        )

        print(f"Saving results as {path}")
        df_out.write_csv(path)
