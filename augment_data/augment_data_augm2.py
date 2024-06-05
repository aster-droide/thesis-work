import librosa
import numpy as np
import os
import audiomentations as am
import soundfile as sf
import random


def augment_audio(file_path_param, output_dir_param, num_augmented=5, original_sr=16000):
    """
    Augment an audio file by applying a random selection of three transformations from
    pitch shifting, time stretching, dynamic range compression, band-pass filtering,
    dynamic range compression, random cuts, frequency masking, and time masking.
    Save the augmented versions in the specified output directory with 'aug_' prefix.
    """
    # Load the file with the specified original sample rate to avoid automatic resampling
    y, sr = librosa.load(file_path_param, sr=original_sr)
    file_name = os.path.basename(file_path_param)

    all_augmentations = [
        am.PitchShift(min_semitones=-0.5, max_semitones=0.5, p=1),
        am.TimeStretch(min_rate=0.95, max_rate=1.05, p=1),
        am.Gain(min_gain_in_db=-3, max_gain_in_db=3, p=1),
        am.BandPassFilter(min_center_freq=300, max_center_freq=3000, p=1),
        am.PolarityInversion(p=0.2),
        am.TimeMask(min_band_part=0.1, max_band_part=0.2, fade=False, p=1)
    ]

    # Select three random augmentations from the list
    selected_augmentations = random.sample(all_augmentations, 3)
    augmenter = am.Compose(selected_augmentations)

    for i in range(num_augmented):
        y_augmented = augmenter(y, sr)
        aug_file_name = file_name.replace('.wav', f'_aug_{i}.wav')
        aug_path = os.path.join(output_dir_param, aug_file_name)
        sf.write(aug_path, y_augmented, sr)


# Apply augmentation to each file in the dataset directory
dataset_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/AugmD2/EverythingAndAllCropped'
output_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/AugmD2/AugmD2-files'

for fn in os.listdir(dataset_dir):
    if fn.endswith('.wav'):
        file_path = os.path.join(dataset_dir, fn)

        cat_class = fn.split('-')[0][:-1]  # remove 'Y' at end

        if 0.5 <= float(cat_class) < 12:
            num_aug = 1  # less augmentation for majority class
        elif float(cat_class) < 0.5:
            num_aug = 4
        else:  # senior
            num_aug = 4  # more augmentation for minority class

        augment_audio(file_path, output_dir, num_augmented=num_aug)
