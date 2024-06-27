from config import Config
from utils.preprocess_utils import get_species_df, run_parallel_preprocessing
import pandas as pd

cfg = Config() # contains the configuration for preprocessing, modify from config.py
species_df = get_species_df(cfg)
species_df.to_csv(cfg.out_df_dir)

test = False

if test:
    # Run preprocessing (for test, run with num_samples=X and out_dir=cfg.temp_dir to save the images to a temporary folder as mentiond in the config)
    img_stats, errors = run_parallel_preprocessing(cfg, data=species_df, out_dir=cfg.temp_dir, num_samples=2)
else:
    # Run preprocessing (for final run, run with num_samples=None, and out_dir=cfg.out_dir to save the images to the out_dir directory mentioned in the config)
    img_stats, errors = run_parallel_preprocessing(cfg, data=species_df, out_dir=cfg.out_dir, num_samples=None)

if len(img_stats):
    img_stats.to_csv(cfg.out_img_stats_dir)