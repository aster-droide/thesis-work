import librosa
import numpy as np
import os
import audiomentations as am
import soundfile as sf


def augment_audio(file_path_param, output_dir_param, pitch_shift_range=(-1.5, 1.5), time_stretch_range=(0.8, 1.2),
                  decibel_range=(-5, 5), num_augmented=5, original_sr=16000):
    """
    Augment an audio file by applying random pitch shifting, time stretching,
    and dynamic range compression, ensuring the output matches the original sample rate.
    Save the augmented versions in the specified output directory with 'aug_' prefix.
    """
    # Load the file with the specified original sample rate to avoid automatic resampling
    y, sr = librosa.load(file_path_param, sr=original_sr)
    file_name = os.path.basename(file_path_param)

    augmenter = am.Compose([
        am.PitchShift(min_semitones=pitch_shift_range[0], max_semitones=pitch_shift_range[1], p=1),
        am.TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1], p=1)
    ])

    for i in range(num_augmented):
        y_augmented = augmenter(y, sr)
        aug_file_name = file_name.replace('.wav', f'_aug_{i}.wav')
        aug_path = os.path.join(output_dir_param, aug_file_name)
        sf.write(aug_path, y_augmented, sr)


# Apply augmentation to each file in the dataset directory
dataset_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/Everything'
output_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/AugmDs/AugmD10-files'

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


