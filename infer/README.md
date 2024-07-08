## Inference Instructions

To perform inference with your trained model, follow these steps after successfully training your classifier as described in the [train_audio](../train_audio) section.

### Running Inference

Use the `infer.py` script to predict species from audio files. This script requires several command line arguments:

- **config**: The configuration filename.
- **weight**: Path to the trained model weights.
- **audio**: Path to the audio file for inference.
- **export**: Directory path where the output CSV files will be saved.

### Example Command
```bash
python3 infer.py --config ait_bird_local --weight ./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt --audio ./data/soundscape_29201.ogg --export ./exports/
```
This command will run inference on the audio file `./data/soundscape_29201.ogg` using the model stored at `./weights/ait_bird_local_eca_nfnet_l0/fold_0_model.pt`. The results will be exported to the `./exports/` directory.

### Output Details
Two CSV files are generated for each audio file:
- **Logits CSV**: Contains the logits for all species, named with the `_logits` suffix.
- **Classification CSV**: Contains the predicted class and score for each segment, named with the `_classification` suffix.

### Understanding the Output
Each row in the output CSVs corresponds to a 5-second segment of the audio data. The `Row_id` column indicates the specific time segment:
- For instance, the suffix `_5` refers to the first 5 seconds of the audio.

### Microservice 
Code for running inference as a microservice to be added soon.

