## Training Instructions for Classifiers

This guide provides instructions for training bird sound classification models. Follow these steps to configure and initiate your training process.

### Step 1: Configuration
Create a configuration file based on the provided [example](./configs/ait_bird_local.py). This file includes settings such as paths to data, model specifications, spectrogram details, and class labels. Modify this file to suit your project needs.

### Step 2: Download Data
Ensure you have all necessary data by downloading from the sources below and storing them according to the paths specified in the config file. 
- **Training Audio**: Download bird audio data from [this link](https://qnap-2.aicenter.dynu.com/share.cgi?ssid=1fb4aa1ecbbc4ea8ac8a2c447e80453b). For additional bird species or missing audio files, visit [xeno-canto](https://xeno-canto.org/). You may also include additional animals by simply adding the audio files into separate folders alongisde the provided data and updating the config file.  
- **Corrupt Audio**: For dealing with corrupt audio, follow the instructions from `first_steps`. You can also download the corrected audio for the 67 species from [this link](). However, if you plan to add more data from xeno-canto or other sources that contain corrupted audio files, you may want to follow the `first_steps`. 
- **Nocall Audio**: Download the nocall samples ('ff1010bird_nocall') from [Kaggle](https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise).
- **Additional Noises**: Download 'esc50' and 'zenodo_nocall_30sec' from [Kaggle](https://www.kaggle.com/datasets/atsunorifujita/birdclef-2023-additional?select=zenodo_nocall_30sec).
- **Pretrained Models**: Download models, e.g., eca_nfnet_l0 from [Kaggle](https://www.kaggle.com/datasets/atsunorifujita/birdclef2023-4th-models), and save them in the `pretrained` folder as specified in your config file.

### Step 3: Prepare Data
Run `create_custom_pkl_df.py` with correct paths to generate a DataFrame pickle file which organizes your dataset in a manner expected by the training script. Update the dataframe path in your config file accordingly. It is expected that you have the audio directories for each species inside the `sound_path` folder. This will generate the ``

## Expected Directory Structure for Given Example Config

```
├── dataframes
│   ├── ait_train_meta.pickle
│   ├── ff1010bird_metadata_v1_pseudo.pickle
│   └── sample_ait_train_meta.pickle
├── external
│   ├── esc50
│   └── zenodo_nocall_30sec
└── sounds
    ├── Abroscopus-superciliaris
    ├── Alcedo-atthis
    ├── Alophoixus-pallidus
    ├── Anthus-hodgsoni
    ├── Apus-cooki
    ├── Bambusicola-fytchii
    ├── nocall
    ├── ... (Other species)
```

### Step 4: Start Training
Execute the training script with the configured settings:
```bash
python3 train.py -C ait_bird_local
```
Replace `ait_bird_local` with your configuration file's name if different.

### Output
- **Logs**: Training logs will be stored in the `logs` directory.
- **Model Weights**: The best model weights will be saved in the `out_weights` directory, or as defined in your config file. These weights are used for inference.

### Note
If training a new model from scratch, consider pretraining your model on large-scale datasets from previous competitions such as the BirdCLEF challenges. Detailed instructions and resources can be found [here](https://github.com/AtsunoriFujita/BirdCLEF-2023-Identify-bird-calls-in-soundscapes). 
