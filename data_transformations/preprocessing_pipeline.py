from common.parameters import DURATION_FOR_TRAINING
from data_structs.track import Track


class PreprocessingPipeline:
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed

    def __call__(self, track: Track, noise_track: Track = None):
        self.unite_duration(track)
        if noise_track is not None:
            track.reduce_noise(noise_track)

    def unite_duration(self, track: Track):
        track.random_subset(DURATION_FOR_TRAINING, self.random_seed)
        track.pad_to_duration(DURATION_FOR_TRAINING)
