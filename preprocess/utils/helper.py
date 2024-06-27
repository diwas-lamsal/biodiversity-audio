import librosa
import numpy as np
import os
import cv2
import pathlib
import pandas as pd

def get_mel_spec_db(cfg, filename, offset):
    """Get dB scaled mel power spectrum"""
    required_len = cfg.seconds * cfg.sample_rate
    sig, dr = librosa.load(path=filename, sr=cfg.sample_rate, offset=(offset * cfg.seconds), duration=cfg.seconds)
    sig = np.concatenate([sig, np.zeros((required_len - len(sig)), dtype=sig.dtype)])
    mel_spec = librosa.feature.melspectrogram(
        y=sig, 
        hop_length=cfg.hop_length,
        sr=cfg.sample_rate, 
        n_fft=cfg.n_fft, 
        n_mels=cfg.n_mels,
        center=cfg.center,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=cfg.top_db)
    return mel_spec_db


def normalize_img(img):
    """Normalize to uint8 image range"""
    assert img.ndim == 2, "unexpected dimension"
    v_min, v_max = np.min(img), np.max(img)
    return ((img - v_min) / (v_max - v_min) * 255).astype('uint8')

def process_record(cfg, out_dir, rec):
    """Process a single record"""
    rec_dir = out_dir + rec.species
    os.makedirs(rec_dir, exist_ok=True)
    stats = []
    base_stat = {"label": rec.label, "filename": rec.filename}
    for offset in range(rec.num_offset):
        mel_spec_db = get_mel_spec_db(cfg, rec.filename, offset=offset)
        img = normalize_img(mel_spec_db)
        fname = f"{pathlib.Path(rec.filename).stem}_{offset}.jpeg"
        path_img = os.path.join(rec_dir, fname)
        ret = cv2.imwrite(path_img, img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        stat = base_stat.copy()
        stat.update({
            "offset": offset,
            "ret": ret,
            "filename": "/".join(pathlib.Path(path_img).parts[-2:]),
        })
        stats.append(stat)
    return pd.DataFrame(stats)


def process_data(cfg, out_dir, data):
    """Process dataframe"""
    errors = []
    l_stats = []
    for rec in data.itertuples():
        try: 
            stats = process_record(cfg, out_dir, rec)
            l_stats.append(stats)
        except Exception as err:
            print(f"Error reading {rec.filename}: {str(err)}")
            errors.append((rec.filename, str(err)))
    return l_stats, errors



