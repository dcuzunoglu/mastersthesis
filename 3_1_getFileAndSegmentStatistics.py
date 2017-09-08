"""
This file is used to create file and segment statistics features from the training and testing recordings.

Adapted and modified from Mario Lasseck's implementation
"""

import numpy as np
import _pickle as pickle

from GlobalStuff import working_directory, audio_files_4_training_dir, audio_files_4_testing_dir
from Util import get_nips4b_metadata, train_file_index_to_train_file_name_without_ext, \
    test_file_index_to_test_file_name_without_ext, create_specgram_image_of_unprocessed_audio_file

# File statistics include min, max, mean and standard deviation taken from all values of the unprocessed
# spectrogram. In addition, the spectrogram is divided into 16 equally sized and distributed frequency bands
# and their min, max, mean and std are also included

# For segment statistics, the number of segments per file plus min, max, mean, and std for width,
# height and frequency position of all segments per file are calculated

# Number of file stats: 68
# Number of segment stats: 13

# File statistics (from all values of the unprocessed spectrogram)
# A total of 68 properties : num of properties for whole file + num of properties per freq band * num of freq bands
num_of_properties_for_whole_file = 4  # Min, Max, Mean, Std
num_of_properties_per_freq_band = 4  # Min, Max Mean, Std
num_of_freq_bands = 16

# Segment statistics
num_of_properties_per_segment = 3  # Width, Height, y-position

# Number of segment properties
# A total of 13 properties NumOfSegments, (Width, Height, PosY) * Min, Max, Mean, Std
num_of_segment_properties = 1 + num_of_properties_per_segment * 4

# Number of file properties
num_of_file_properties = num_of_properties_for_whole_file + num_of_properties_per_freq_band * num_of_freq_bands


def get_file_properties():
    """
    Create an array the size of number of file properties, and fill the array with file properties
    :return:
    """
    # Create an array the size of number of file properties, and fill the
    file_properties_ = np.zeros(num_of_file_properties)
    
    file_properties_[0] = np.min(specgram_image)
    file_properties_[1] = np.max(specgram_image)
    file_properties_[2] = np.mean(specgram_image)
    file_properties_[3] = np.std(specgram_image)

    rows_per_freq_band = int(specgram_image.shape[0] / 16)

    for fb in range(num_of_freq_bands):
        current_freq_band = specgram_image[fb * rows_per_freq_band:fb * rows_per_freq_band + rows_per_freq_band, :]
        minimum = np.min(current_freq_band)
        maximum = np.max(current_freq_band)
        mean = np.mean(current_freq_band)
        std = np.std(current_freq_band)

        file_properties_[num_of_properties_for_whole_file
                         + fb * num_of_properties_per_freq_band:num_of_properties_for_whole_file + fb
                         * num_of_properties_per_freq_band + num_of_properties_per_freq_band] \
            = [minimum, maximum, mean, std]
        
    return file_properties_


def get_segment_properties():
    """
    Create an array the size of number of segment properties, and fill the array with segment properties
    :return:
    """
    segment_properties_ = np.zeros((1 + num_of_properties_per_segment * 4))
    segment_properties_[0] = num_of_segments
    
    properties_per_segment = np.empty((num_of_properties_per_segment, num_of_segments))

    for s in range(num_of_segments):
        [x_min, x_max, y_min, y_max] = segment_metadata_of_current_file[s][0]
        
        properties_per_segment[0][s] = x_max-x_min   # Width
        properties_per_segment[1][s] = y_max-y_min   # Height
        properties_per_segment[2][s] = y_min         # PosY
        
    if num_of_segments > 0:
        segment_properties_[1:4] = np.min(properties_per_segment, axis=1)
        segment_properties_[4:7] = np.max(properties_per_segment, axis=1)
        segment_properties_[7:10] = np.mean(properties_per_segment, axis=1)
        segment_properties_[10:13] = np.std(properties_per_segment, axis=1)

    return segment_properties_


# ============================= MAIN ======================================
# Get metadata from files
num_of_train_files, num_of_bird_classes, bird_classes_per_train_index, \
    train_indices_with_bird_classes, train_indices_ass_with_bird_class = get_nips4b_metadata()

# ============ Train
print("\nProcessing TrainFiles ...")

audio_files_dir = audio_files_4_training_dir

# A total of 81 file and segment properties exist
# Create an empty array of size num train files*num file and segment properties
num_of_file_and_segment_properties = num_of_file_properties + num_of_segment_properties
file_and_segment_properties_train = np.empty((num_of_train_files, num_of_file_and_segment_properties))

# Import segment metadata of training files
pkl_file = open(working_directory + 'TrainData/_SegmentMetaData.pkl', 'rb')
segments_meta_data = pickle.load(pkl_file)
pkl_file.close()

# For each training file
for ti in range(num_of_train_files):

    # Get segment metadata of current training file
    segment_metadata_of_current_file = segments_meta_data[ti][0]
    num_of_segments = len(segment_metadata_of_current_file)

    # Get the train file name from test file index
    segmented_filename_without_ext = train_file_index_to_train_file_name_without_ext(ti)
    file_name = segmented_filename_without_ext + '.wav'

    # Create unnormalized spectrogram of the current test file
    # and get the height and width of spectrogram image
    specgram_image = create_specgram_image_of_unprocessed_audio_file(audio_files_dir + file_name)
    specgram_image_height, specgram_image_width = specgram_image.shape

    # Get file and segment properties defined above
    file_properties = get_file_properties()
    segment_properties = get_segment_properties()

    # Record file and segment properties of each file in an array
    file_and_segment_properties_train[ti][:] = np.hstack((file_properties, segment_properties))

    print('File: ', ti)

# ================ Test

num_of_test_files = 1000

print("\nProcessing TestFiles ...")

audio_files_dir = audio_files_4_testing_dir

# A total of 81 file and segment properties exist
# Create an empty array of size num test files*num file and segment properties
num_of_file_and_segment_properties = num_of_file_properties + num_of_segment_properties
file_and_segment_properties_test = np.empty((num_of_test_files, num_of_file_and_segment_properties))

# Import segment metadata of test files
pkl_file = open(working_directory + 'TestData/_SegmentMetaData.pkl', 'rb')
segments_meta_data = pickle.load(pkl_file)
pkl_file.close()

# For each test file
for ti in range(num_of_test_files):

    # Get segment metadata of current test file
    segment_metadata_of_current_file = segments_meta_data[ti][0]
    num_of_segments = len(segment_metadata_of_current_file)

    # Get the test file name from test file index
    segmented_filename_without_ext = test_file_index_to_test_file_name_without_ext(ti)
    file_name = segmented_filename_without_ext + '.wav'

    # Create unnormalized spectrogram of the current test file
    # and get the height and width of spectrogram image
    specgram_image = create_specgram_image_of_unprocessed_audio_file(audio_files_dir + file_name)
    specgram_image_height, specgram_image_width = specgram_image.shape  # print SpecgramImage.shape

    # Get file and segment properties defined above
    file_properties = get_file_properties()
    segment_properties = get_segment_properties()

    # Record file and segment properties of each file in an array
    file_and_segment_properties_test[ti][:] = np.hstack((file_properties, segment_properties))

    print('File:', ti)

# ==================== Scaling
# Scale the properties into range [0, 1]
print("\nPostProcessing (Scaling Feature Rows --> 0..1)")

print('Min/Max org: ', np.min(file_and_segment_properties_train), np.max(file_and_segment_properties_train))

file_and_segment_properties_train_row_min = np.min(file_and_segment_properties_train, axis=0)
file_and_segment_properties_train_row_max = np.max(file_and_segment_properties_train, axis=0)

file_and_segment_properties_train_row_scaled = \
    (file_and_segment_properties_train - file_and_segment_properties_train_row_min) \
    / file_and_segment_properties_train_row_max
file_and_segment_properties_test_row_scaled = \
    (file_and_segment_properties_test - file_and_segment_properties_train_row_min) \
    / file_and_segment_properties_train_row_max

print('Min/Max after: ', np.min(file_and_segment_properties_train_row_scaled),
      np.max(file_and_segment_properties_train_row_scaled))

print('FileAndSegmentStatistics4TestSet shape: ', file_and_segment_properties_test_row_scaled.shape)

output = open(working_directory + 'TrainData/_FileAndSegmentStatisticsScaled.pkl', 'wb')
pickle.dump(file_and_segment_properties_train_row_scaled, output)
output.close()  

output = open(working_directory + 'TestData/_FileAndSegmentStatisticsScaled.pkl', 'wb')
pickle.dump(file_and_segment_properties_test_row_scaled, output)
output.close()
