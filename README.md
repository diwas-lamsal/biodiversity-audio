# Biodiversity Audio Classification

Biodiversity Audio Classification repository. This project is dedicated to training machine learning models to classify audio recordings of wildlife, with a specific focus on bird species from Chiang Mai, Thailand. It is possible to expand the classifier to more bird species or other animals based on their sounds as heard in their natural habitat. 

## Quick Start with Docker

We provide a Docker environment that encapsulates all necessary dependencies. Find it in the [`docker folder`](./docker/).

## Dataset

To train models with the provided code:
- **Download Bird Data**: [Access the dataset here](https://qnap-2.aicenter.dynu.com/share.cgi?ssid=1fb4aa1ecbbc4ea8ac8a2c447e80453b).

### Note on Missing Species
Due to restrictions on xeno-canto, data for the following species are not included:
- **Copsychus malabaricus**: [Species Info](https://xeno-canto.org/species/Copsychus-malabaricus)
- **Copsychus saularis**: [Species Info](https://xeno-canto.org/species/Copsychus-saularis)

These may be downloaded directly from [xeno-canto](https://xeno-canto.org/) using their [API](https://github.com/ntivirikin/xeno-canto-py).

## Handling Corrupt Audio Data

Some audio files may be corrupted. To address this issue, instructions for re-encoding the audio using ffmpeg are provided in the `first_steps` folder.

## Training and Inference

- **Training**: Follow the detailed instructions in the `train_audio` folder.
- **Inference**: Guidelines are provided in the `infer` folder to help you run models and make predictions.