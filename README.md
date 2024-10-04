# Biodiversity Audio Classification

Biodiversity Audio Classification repository. This project is dedicated to training machine learning models to classify audio recordings of wildlife, with a specific focus on bird species from Chiang Mai, Thailand. It is possible to classify more bird species or other animals based on their sounds as heard in their natural habitat. 

The codebase is adapted from [BirdCLEF 2023 4th Place Solution](https://www.kaggle.com/competitions/birdclef-2023/discussion/412753). We train without knowledge distillation and only for a single fold for simplicity. 

## Quick Start with Docker

We provide a Docker environment that encapsulates all necessary dependencies. Find it in the [`docker folder`](./docker/).

## Dataset

To train models with the provided code:
- **Download Bird Data**: [Access the dataset here](https://qnap-2.aicenter.dynu.com/share.cgi?ssid=1fb4aa1ecbbc4ea8ac8a2c447e80453b).
- **Preprocessing Corrupt Audio**: For a lot of audio files downloaded directly from xeno-canto, there is a possibility that the audio files may be corrupt due to bad header files or some other reason. To address this issue, instructions for re-encoding the audio using ffmpeg are provided in the [`first_steps`](./first_steps/) folder.

### Note on Missing Species
Due to restrictions on xeno-canto, data for the following species are not included at the time of writing:
- **Copsychus malabaricus**: [Species Info](https://xeno-canto.org/species/Copsychus-malabaricus)
- **Copsychus saularis**: [Species Info](https://xeno-canto.org/species/Copsychus-saularis)

If available in the future, these may be downloaded directly from [xeno-canto](https://xeno-canto.org/) or using an [API](https://github.com/ntivirikin/xeno-canto-py).


## Training and Inference

- **Training**: Follow the detailed instructions in the [`train_audio`](./train_audio/) folder.
- **Inference**: Guidelines are provided in the [`infer`](./infer/) folder to help you run models and make predictions.

## Install Flake8 linter
1. Run `pip install pre-commit`
2. Check if pre-commit is installed `pre-commit --version`
3. Run `pre-commit install`

This will block commit if there is error in staged files.
