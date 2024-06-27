from config import Config
from utils.train_utils import run_training, get_train_test_split
import pandas as pd
import tensorflow as tf

cfg = Config() # contains the configuration for training, modify from config.py

data = pd.read_csv(cfg.data_stats_path)
data["path_img"] = cfg.dataset_dir + data["filename"]

train_df, valid_df = get_train_test_split(cfg, data)
hist, oof = run_training(cfg, train_df, valid_df, cfg.model_name)
oof.to_csv("oof.csv")