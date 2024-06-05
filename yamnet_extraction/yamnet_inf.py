import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import librosa
import re
import os
import pandas as pd


def extract_age_from_filename(filename):
    """Extracts the age part from the filename."""
    # Use regex to find the age pattern (e.g., "0.5Y" or "0Y")
    match = re.match(r"(\d+\.?\d*)Y", filename)
    if match:
        # Convert the matched age to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, return None or raise an error
        return None


def extract_pitch_from_filename(filename):
    """
    Extracts the pitch (mean F0) from the filename.

    The function assumes the pitch is at the end of the filename,
    preceded by a hyphen, and followed by the '.wav' extension.

    Parameters:
    - filename: The name of the file, as a string.

    Returns:
    - The extracted pitch as a float, or None if not found.
    """
    # Regex pattern to match a hyphen followed by one or more digits and potentially a decimal point
    pattern = re.compile(r'-(\d+\.\d+|\d+)\.wav$')
    match = pattern.search(filename)
    if match:
        # Convert the matched pitch to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, return None
        return None


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


def extract_gender_from_filename(filename):
    """Extracts the gender part from the filename."""
    # Adjust regex to find the gender indicator more accurately
    match = re.search(r"([MFX])(?=-|\d|\.wav)", filename)
    if match:
        # Return the matched gender
        return match.group(1)
    else:
        # if gender not documented in filename return 'X' (UNKNOWN)
        return "X"


# Directory containing the audio files
audio_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/AugmDs/Perch/AugmD19-looped'

# List of audio files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Load the model.
model = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')

data_list = []
for audio_file in audio_files:

    print(f"Processing {audio_file}")

    # Extract the filename from the full path
    filename = os.path.basename(audio_file)

    # Regex pattern to extract cat identifiers
    pattern = re.compile(r'-(\d{3}[A-Z])')
    # Search for the pattern in the filename
    match = pattern.search(filename)
    if match:
        # Extract cat identifier
        cat_id = match.group(1)
        print(cat_id)
    else:
        raise ValueError("Identifier missing or incorrect: ", filename)

    # Extract the target class from the filename
    age = extract_age_from_filename(filename)
    print(age)
    target_class = age

    # Full path to the audio file
    full_audio_path = os.path.join(audio_dir, audio_file)

    # Load the audio file using librosa
    waveform, sample_rate = librosa.load(full_audio_path, sr=16000, mono=True)

    # Scale and convert the waveform to the expected format.
    waveform = waveform.astype(np.float32)

    print(waveform.shape)

    # Run the model, check the output.
    scores, embeddings, log_mel_spectrogram = model(waveform)

    """
    testing
    """
    scores.shape.assert_is_compatible_with([None, 521])
    embeddings.shape.assert_is_compatible_with([None, 1024])
    log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

    # print(scores)
    # print(embeddings)

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
    print(class_names[scores.numpy().mean(axis=0).argmax()])
    """
    testing
    """

    # Extract the gender from the filename
    gender = extract_gender_from_filename(filename)
    print(gender)
    gender_class = gender

    # # Extract pitch from filename (generated with crepe)
    # mean_freq = extract_pitch_from_filename(filename)

    # Append the embeddings, mean frequency, target class, and cat identifier to the data list
    for embedding in embeddings.numpy():
        data_list.append([embedding, gender_class, target_class, cat_id])

# Create a DataFrame with embeddings and corresponding labels
embeddings_df = pd.DataFrame(data_list, columns=['embedding', 'gender', 'target', 'cat_id'])

# Expand 'embedding' column to separate columns
embeddings_df = pd.concat([pd.DataFrame(embeddings_df['embedding'].tolist()),
                           embeddings_df['gender'],
                           embeddings_df['target'],
                           embeddings_df['cat_id']], axis=1)

# Convert all column names to strings
embeddings_df.columns = embeddings_df.columns.astype(str)

# Save the DataFrame to a CSV file
embeddings_df.to_csv('yamnet_embeddings.csv', index=False)



