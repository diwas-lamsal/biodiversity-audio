import torch
import pandas as pd
import numpy as np
import librosa
import os

class BirdClef2023Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str = 'DATA_PATH',
        period: float = 5.0,
        secondary_coef: float = 1.0,
        smooth_label: float = 0.05,
        df: pd.DataFrame = 'DATAFRAME',
        train: bool = True,
        tr_transforms: list = [],
        cfg = None,
    ):

        self.df = df
        self.data_path = data_path
        self.filenames = df["filename"]

        self.primary_label = df["primary_label"]
        self.secondary_labels = (
            df["secondary_labels"]
            .map(
                lambda s: s.replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("'", "")
                .split(" ")
            ).values
        )

        self.secondary_coef = secondary_coef
        self.type = df["type"]
        self.rating = df["rating"]
        self.period = period
        self.smooth_label = smooth_label + 1e-6
        self.wave_transforms = tr_transforms
        self.train = train
        self.cfg = cfg

    def __len__(self):
        return len(self.df)
    
    
    def load_wave_and_crop(self, filename, period, start=None):

        waveform_orig, sample_rate = librosa.load(filename, sr=32000, mono=False)
        
        # Check if the waveform is stereo (two channels)
        if waveform_orig.ndim == 2:
            # If stereo, you can average the channels to create a mono waveform
            waveform_orig = librosa.to_mono(waveform_orig)

        wave_len = len(waveform_orig)
        waveform = np.concatenate([waveform_orig, waveform_orig, waveform_orig])

        effective_length = sample_rate * period
        
        while len(waveform) < (period * sample_rate * 3):
            waveform = np.concatenate([waveform, waveform_orig])
        if start is not None:
            start = start - (period - 5) / 2 * sample_rate
            while start < 0:
                start += wave_len
            start = int(start)
        else:
            if wave_len < effective_length:
                start = np.random.randint(effective_length - wave_len)
            elif wave_len > effective_length:
                start = np.random.randint(wave_len - effective_length)
            elif wave_len == effective_length:
                start = 0

        waveform_seg = waveform[start: start + int(effective_length)]

        return waveform_orig, waveform_seg, sample_rate, start



    def __getitem__(self, idx):

        filename = os.path.join(self.data_path, self.filenames[idx])

        if self.train:
            waveform, waveform_seg, sample_rate, start = self.load_wave_and_crop(
                filename, self.period
            )
            waveform_seg = self.wave_transforms(
                samples=waveform_seg, sample_rate=sample_rate
                )
        else:
            waveform, waveform_seg, sample_rate, start = self.load_wave_and_crop(
                filename, self.period, 0
            )

        waveform_seg = torch.from_numpy(np.nan_to_num(waveform_seg)).float()

        rating = self.rating[idx]

        target = np.zeros(self.cfg.num_classes, dtype=np.float32)
        if self.primary_label[idx] != 'nocall':
            primary_label = self.cfg.bird2id[self.primary_label[idx]]
            target[primary_label] = 1.0

            if self.train:
                for s in self.secondary_labels[idx]:
                    if s != "" and s in self.cfg.bird2id.keys():
                        target[self.cfg.bird2id[s]] = self.secondary_coef

        target = torch.from_numpy(target).float()

        return {
            "wave": waveform_seg,
            "rating": rating,
            "primary_targets": (target > 0.5).float(),
            "loss_target": target * (1-self.smooth_label) + self.smooth_label / target.size(-1),
        }

