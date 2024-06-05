import os
import librosa
import soundfile as sf
import random


def random_cut_and_save(file_path, output_dir, min_percentage=45, max_percentage=70):
    """
    Load an audio file, cut a random percentage of its duration, and save it to the output directory.
    Only cuts if the audio is longer than 0.7 seconds.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate
    duration = librosa.get_duration(y=y, sr=sr)

    if duration > 0.7:
        # Calculate the percentage to cut
        cut_percentage = random.uniform(min_percentage, max_percentage) / 100.0
        cut_duration = duration * cut_percentage
        start_time = random.uniform(0, duration - cut_duration)
        start_sample = int(start_time * sr)
        end_sample = int(start_sample + cut_duration * sr)

        # Extract the segment
        y_cut = y[start_sample:end_sample]

        # Prepare the output path and save the file
        base_name = os.path.basename(file_path)
        base_name_without_ext = os.path.splitext(base_name)[0]  # Remove the extension
        output_filename = f"{base_name_without_ext}-cut.wav"  # Append '-cut' and add back the extension
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, y_cut, sr)


def process_directory(input_dir, output_dir):
    """
    Process all .wav files in the input directory, applying random cuts and saving to the output directory.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            random_cut_and_save(file_path, output_dir)


# Set the paths for your directories
input_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/Everything'
output_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/AugmD2/AllCropped'

# Process the directory
process_directory(input_dir, output_dir)
