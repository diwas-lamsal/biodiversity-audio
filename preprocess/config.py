from pydantic import BaseModel as ConfigBaseModel
class Config(ConfigBaseModel):
    # data
    # sound_dir = "../../../Dataset/AIC_BirdDataset/sounds/"    
    sound_dir = "./data/sounds/"    
    sample_rate = 32_000
    # spec
    img_size = (128, 256)
    seconds = 5
    num_offset_max = 24
    min_duration = 0.5
    n_fft = 2048
    n_mels = img_size[0]
    hop_length = (seconds * sample_rate - n_fft) // (img_size[1] - 1) 
    center = False
    fmin = 500
    fmax = 12_500
    top_db = 80
    # output
    out_dir = "./data/images/"
    out_df_dir = "./data/species_df.csv"
    out_img_stats_dir = "./data/img_stats.csv"
    temp_dir = "./data/temp/"
    jpeg_quality = 100
    cpu_count = "max" # can be a number or max, where max will use the full number of CPUs for preprocessing
