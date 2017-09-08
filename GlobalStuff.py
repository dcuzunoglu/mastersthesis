"""
This file contains the variables that are used throughout all the scripts.

Adapted and modified from Mario Lasseck's implementation
"""

import os

# ==================================================================================

# needs to be adjusted to point to the data provided by NIPS4B 
# Download from:
# https://www.kaggle.com/c/multilabel-bird-species-classification-nips2013/data

train_labels_dir = '/home/doruk/Desktop/Birds/Dataset/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/'
audio_files_4_training_dir = '/home/doruk/Desktop/Birds/Dataset/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/'
audio_files_4_testing_dir = '/home/doruk/Desktop/Birds/Dataset/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/test/'

working_directory = '/home/doruk/Desktop/thesis_repo//outputs/'

# ========================================================================================

FS = 22050.0
FN = FS/2.0
FFT_SIZE = 512
FFT_HOP_SIZE = 128

num_of_low_freq_bins_skipped = 4  # 3 --> 129.2 Hz , 4 --> 172.3 Hz
num_of_high_freq_bins_skipped = 24  # 24 --> 1033.59375 Hz (fMax = 9991.4 Hz)

train_data_directory = working_directory + 'TrainData/'
test_data_directory = working_directory + 'TestData/'
submission_directory = working_directory + 'Submission/'

if not os.path.exists(working_directory):
    os.makedirs(working_directory)

if not os.path.exists(train_data_directory):
    os.makedirs(train_data_directory)

if not os.path.exists(test_data_directory):
    os.makedirs(test_data_directory)

if not os.path.exists(submission_directory):
    os.makedirs(submission_directory)
