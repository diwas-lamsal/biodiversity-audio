from pydantic import BaseModel as ConfigBaseModel
from datetime import datetime

class Config(ConfigBaseModel):
    ## general
    run_ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    debug = False
    model_name = "mobilenet"
    test_size = 0.2
    seed = 42
    fit_verbose = 2
    ## data
    dataset_dir = "./data/images/"
    data_stats_path = "./data/img_stats.csv"
    label = "label"
    n_label = 67
    img_size = (128, 256) # Same as defined in the preprocess step
    channels = 1
    img_shape = (*img_size, channels)
    ## model
    base_model_weights = "imagenet"
    dropout = 0.20
    ## training
    label_smoothing = 0.05
    shuffle_size = 1028
    steps_per_epoch = 300
    batch_size = 64  
    valid_batch_size = batch_size
    epochs = 50
    patience = 4
    monitor = "val_loss"  # val_loss
    monitor_mode = "auto"
    lr = 1e-3
    ## aug
    aug_proba = 0.8
