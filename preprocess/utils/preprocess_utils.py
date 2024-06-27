import os
import pandas as pd
import numpy as np
import librosa
from joblib import delayed, Parallel
from utils.helper import process_data
import pathlib

def get_duration(rec):
    return librosa.get_duration(path=rec["filename"])

def get_duration_df(df):
    return df.apply(get_duration, axis=1)

def get_species_df(cfg):
    species = []
    file_paths = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(cfg.sound_dir):
        # Get the species name from the directory name
        current_species = os.path.basename(dirpath)
        for filename in filenames:
            if filename.endswith('.mp3'):  # Filter for mp3 files
                # Full path to the file
                full_path = os.path.join(dirpath, filename)
                # Append data to lists
                species.append(current_species)
                file_paths.append(full_path)
                
    data = {'species': species, 'filename': file_paths}
    data = pd.DataFrame(data)
    
    # The code expects the sound directory to contain the folders for each species, including the non-sound or background event
    labels = sorted(os.listdir(cfg.sound_dir))
    assert labels == sorted(labels), "labels are not sorted"
    label_encoder = pd.Series(np.arange(len(labels)), index=labels)
    data["label"] = data["species"].map(label_encoder)
    
    durations = Parallel(n_jobs=os.cpu_count(), verbose=1, backend='multiprocessing')(
        delayed(get_duration_df)(sub) 
        for sub in np.array_split(data, os.cpu_count())
    )
    data["duration"] = pd.concat(durations)
    
    data["num_offset"] = (1 + (data["duration"] - cfg.min_duration) // cfg.seconds).astype('int')
    data["num_offset"] = data["num_offset"].clip(upper=cfg.num_offset_max)
    
    return data


def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def run_parallel_preprocessing(cfg, data, out_dir, num_samples=None):
    num_cpus = os.cpu_count() if cfg.cpu_count =="max" else num_cpus
    data = data if not num_samples else data.sample(num_samples)
    results = Parallel(n_jobs=os.cpu_count(), verbose=1, backend='multiprocessing')(
        delayed(process_data)(cfg, out_dir, sub) for sub in np.array_split(data, os.cpu_count())
    )
    errors = [x for r in results for x in r[1]]
    img_stats = [x for r in results for x in r[0]]
    if len(img_stats):
        img_stats = pd.concat(img_stats).reset_index(drop=True)
        
    print("Expected number of images:", data["num_offset"].sum())
    print("Actual exported images:", len(img_stats))
    print("Total errors:", len(errors))
    
    bs = sum(os.stat(f).st_size for f in pathlib.Path(out_dir).glob("*/*"))
    print(f"Total size of {out_dir} after preprocessing: {convert_bytes(bs)}")
    
    return img_stats, errors


        