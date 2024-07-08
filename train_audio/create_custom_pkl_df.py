import os
import pandas as pd
import pickle

# Define the path to the bird sound files
save_path = './data/dataframes'
sound_path = './data/sounds'

# List of all bird species directories
bird_species = os.listdir(sound_path)

# Define a dictionary for ratings
ratings_map = {'A': 5.0, 'B': 4.0, 'C': 3.0, 'D': 2.0, 'E': 1.0}

# Prepare lists to hold data
data = {
    'primary_label': [],
    'secondary_labels': [],
    'scientific_name': [],
    'rating': [],
    'filename': [],
    'fold': [],
    'type': [],
}

# Process each bird species folder
for species in bird_species:
    species_path = os.path.join(sound_path, species)
    if os.path.exists(species_path):
        for file in os.listdir(species_path):
            if file.endswith('.mp3'):
                # Extract rating from filename
                file_suffix = file.split('-')[-1][1]  # Expecting format like 'XC629222 - Species - B.mp3'
                rating = ratings_map.get(file_suffix, 1.0)
                
                # Add data to lists
                data['primary_label'].append(species)
                data['secondary_labels'].append("[]")
                data['scientific_name'].append(species.replace('-', ' '))
                data['rating'].append(rating)
                data['filename'].append(f"{species}/{file}")
                data['fold'].append(0)  # All folds set to 0 as specified
                data['type'].append('[]')  # All folds set to 0 as specified

# Create DataFrame
df = pd.DataFrame(data)

output_path = os.path.join(save_path, "ait_train_meta.pickle")
try:
    # Save the DataFrame as a pickle file
    df.to_pickle(output_path)
    print(f"DataFrame successfully saved to {output_path}")
except Exception as e:
    print(f"An error occurred while saving the DataFrame: {e}")