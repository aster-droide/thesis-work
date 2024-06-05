import crepe
import os
from scipy.io import wavfile
import shutil


# Directory containing the audio files
audio_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/catmeow-dataset-simone-trimmed-as-is'
new_audio_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/cat-meow-simone-pitched'

# Ensure the new directory exists
os.makedirs(new_audio_dir, exist_ok=True)

# Confidence threshold for including a pitch in the average
high_confidence_threshold = 0.90
mid_confidence_threshold = 0.75
low_confidence_threshold = 0.50

# List of audio files
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

for audio_file in audio_files:
    print(f"Processing {audio_file}")

    # Full path to the audio file
    full_audio_path = os.path.join(audio_dir, audio_file)

    # extract pitch
    sr, audio = wavfile.read(full_audio_path)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, model_capacity="full")

    # Filter by confidence and calculate averages
    filtered_frequencies = [freq for freq, conf in zip(frequency, confidence) if conf >= high_confidence_threshold]
    # Check if any frequencies are above the threshold
    if filtered_frequencies:
        mean_freq = sum(filtered_frequencies) / len(filtered_frequencies)
    # If none are above the high threshold, try the mid threshold
    elif any(conf >= mid_confidence_threshold for conf in confidence):
        filtered_frequencies = [freq for freq, conf in zip(frequency, confidence) if
                                conf >= mid_confidence_threshold]
        mean_freq = sum(filtered_frequencies) / len(filtered_frequencies)

    # If none are above the high threshold, try the lowest threshold
    elif any(conf >= low_confidence_threshold for conf in confidence):
        filtered_frequencies = [freq for freq, conf in zip(frequency, confidence) if
                                conf >= low_confidence_threshold]
        mean_freq = sum(filtered_frequencies) / len(filtered_frequencies)

    else:
        # Use all frequencies if none are above the threshold
        mean_freq = sum(frequency) / len(frequency)

    # Construct new filename with mean F0 value appended
    new_filename = f"{audio_file.rstrip('.wav')}-{mean_freq:.2f}.wav"
    new_full_audio_path = os.path.join(new_audio_dir, new_filename)

    # Copy the file to the new directory with the new name
    shutil.copy(full_audio_path, new_full_audio_path)
    print(f"Copied to {new_full_audio_path}")

print("Pitch extraction and file copying is complete.")
