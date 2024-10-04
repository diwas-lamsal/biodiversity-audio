import audiomentations as AA
import torch_audiomentations as TAA


def get_transformations(cfg):
    tr_transforms = AA.Compose(
        [
            AA.OneOf([
                AA.Gain(min_gain_in_db=-15, max_gain_in_db=15, p=1.0),
                AA.GainTransition(
                    min_gain_in_db=-24.0,
                    max_gain_in_db=6.0,
                    min_duration=0.2,
                    max_duration=6.0,
                    p=1.0
                )
            ], p=0.5,),
            AA.OneOf([
                AA.AddGaussianNoise(p=1.0),
                AA.AddGaussianSNR(p=1.0),
            ], p=0.3,),
            AA.OneOf([
                AA.AddShortNoises(
                    sounds_path=cfg.short_noise_dir,
                    min_snr_in_db=0,
                    max_snr_in_db=3,
                    p=1.0,
                    lru_cache_size=10,
                    min_time_between_sounds=4.0,
                    max_time_between_sounds=16.0,
                ),
            ], p=0.5,),
            AA.OneOf([
                AA.AddBackgroundNoise(
                    sounds_path=cfg.background_noise_dir,
                    min_snr_in_db=0,
                    max_snr_in_db=3,
                    p=1.0,
                    lru_cache_size=1400,),
            ], p=0.5,),
            AA.LowPassFilter(p=0.5),
        ]
    )

    taa_augmentation = TAA.Compose(
        transforms=[
            TAA.PitchShift(
                sample_rate=cfg.sample_rate,
                mode="per_example",
                p=0.2,
                ),
        ]
    )

    return tr_transforms, taa_augmentation
