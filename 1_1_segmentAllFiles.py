"""
This file is used to segment the training and testing recordings

Adapted and modified from Mario Lasseck's implementation
"""

import numpy as np
import scipy as sp

import time
import datetime
import matplotlib.pyplot as plt

from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_closing, binary_dilation
from scipy.ndimage.filters import median_filter

import _pickle as pickle
import os

from GlobalStuff import num_of_low_freq_bins_skipped, num_of_high_freq_bins_skipped, \
    FFT_HOP_SIZE, FFT_SIZE, FS, FN, working_directory, audio_files_4_training_dir, audio_files_4_testing_dir
from Util import create_specgram_image_of_audio_file, get_rgb_value, get_nips4b_metadata, \
    train_file_index_to_train_file_name_without_ext, process_image_for_template_matching, \
    test_file_index_to_test_file_name_without_ext


SaveImagesOfSegmentedSpecgramsFlag = 1


def get_log_image(image):
    return 20.0 * np.log10(image + 0.000000001)


def visualize_image_preprocessing(images_to_show):
    """
    Visualize and save the preprocessed images
    :param images_to_show:
    :return:
    """
    num_of_images_to_show = len(images_to_show)
    fig_pre_processing, (ax1) = plt.subplots(ncols=1, figsize=(10, 15))
    for i in range(num_of_images_to_show):
        plt.subplot(num_of_images_to_show, 1, i+1)
        plt.imshow(images_to_show[i], cmap=plt.cm.gray_r)  # @UndefinedVariable
        plt.xticks([], [])
        plt.yticks([], [])
        
        if i == 0:
            plt.title('Spectrogram', fontsize=18)
        if i == 1:
            plt.title('Median Clipping', fontsize=18)
        if i == 2:
            plt.title('Closing & Dilation', fontsize=18)
        if i == 3:
            plt.title('Median Filter & Small Objects Removed', fontsize=18)

    fig_pre_processing.savefig(
        working_directory + file_type + 'Data/ImagesOfPreProcessing/'
        + segmented_filename_without_ext + '.png', dpi=100)
    fig_pre_processing.clear()
    fig_pre_processing.clf()
    plt.close()


def save_images_of_segmented_specgrams():
    """
    Creates a log spectrogram with labeled segments
    :return:
    """

    fig_specgram_with_segments, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
    specgram_image_cut_log = 20.0 * np.log10(specgram_image_cut + 0.000000001)
    ax1.imshow(specgram_image_cut_log, cmap=plt.cm.gray_r)  # @UndefinedVariable
    # axis labeling
    x_max_in_sec = specgram_image_cut_log.shape[1]*FFT_HOP_SIZE/FS
    x_vec_in_sec = np.arange(0.5, x_max_in_sec, 0.5)  # [0.5, 1.0, 1.5, ...xMaxInSe[
    x_locations = x_vec_in_sec*FS/FFT_HOP_SIZE
    plt.xticks(x_locations, x_vec_in_sec)
    y_vec_in_hz = np.arange(2000, 10000, 2000)
    y_locations = (-(y_vec_in_hz-FN)*FFT_SIZE/FS) - num_of_high_freq_bins_skipped
    plt.yticks(y_locations, y_vec_in_hz/1000)  # [2, 4, 6, 8]
    ax1.set_title('Segmented Spectrogram (log)', fontsize=12)
    plt.xlabel('Time  [s]', fontsize=11)
    plt.ylabel('Frequency  [kHz]', fontsize=11)
    
    cur_segment_index = total_num_of_segments
     
    for s in range(num_of_segments):
        [x_min, x_max, y_min, y_max] = segment_metadata_per_file[s][0]
        rect = plt.Rectangle((x_min, y_min), x_max + 1 - x_min, y_max + 1 - y_min,
                             edgecolor=get_rgb_value('jet', s, num_of_segments), facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x_min, y_min-2, str(cur_segment_index), color='yellow', fontsize=8)
        cur_segment_index += 1

    fig_specgram_with_segments.savefig(
        working_directory + file_type + 'Data/ImagesOfSegmentation/' + segmented_filename_without_ext + '.png', dpi=100)
    fig_specgram_with_segments.clear()
    fig_specgram_with_segments.clf()
    plt.close()
     
    # ======================================================================================================


def remove_noise_per_freqband_and_time_frame(specgram_image):
    """
    Reduce background noise by median clipping: setting pixel values to 1 only if the frequency band and
    time frame values are higher than 3 times the median
    :param specgram_image:
    :return:
    """
    # Remove all pixels that are below 3 times the median of its corresponding row (frequency band)
    specgram_image_noise_per_freq_band_removed = np.empty(specgram_image.shape, dtype=np.float64)
    for i in range(specgram_image.shape[0]):
        specgram_image_noise_per_freq_band_removed[i, :] = specgram_image[i, :]
        low_values_indices = specgram_image[i, :] < 3.0 * np.median(specgram_image[i, :])
        specgram_image_noise_per_freq_band_removed[i, low_values_indices] = 0.0

    # Remove all pixels that are below 3 times the median of its corresponding column (time frame)
    specgram_image_noise_per_time_frame_removed = np.empty(specgram_image.shape, dtype=np.float64)
    for i in range(specgram_image.shape[1]):
        specgram_image_noise_per_time_frame_removed[:, i] = specgram_image[:, i]
        low_values_indices = specgram_image[:, i] < 3.0 * np.median(specgram_image[:, i])
        specgram_image_noise_per_time_frame_removed[low_values_indices, i] = 0.0

    # Pixels equal to 1 only if both the time frame and frequency band is greater than zero
    return np.logical_and(specgram_image_noise_per_freq_band_removed, specgram_image_noise_per_time_frame_removed)


def label_specgram(specgram_image_for_labeling, specgram_image_processed_for_template_matching):
    """
    Returns metadata of segments (x and y positions of borders to segments)
    and segmented spectrograms
    :param specgram_image_for_labeling:
    :param specgram_image_processed_for_template_matching:
    :return:
    """
    segment_border_in_pixel = 12
    
    height = specgram_image_for_labeling.shape[0]
    width = specgram_image_for_labeling.shape[1]

    # Label segments in the spectrogram (each contiguous nonzero area gets a number label each pixel in each area has
    # the same number)
    labeled_segments, num_of_segments_ = sp.ndimage.label(specgram_image_for_labeling)
    
    segment_meta_data_per_file = []
    segment_data_per_file = []

    # Find the border pixels around the segments to form a box
    for current_segment_id in range(num_of_segments_):
        current_segment = (labeled_segments == current_segment_id+1)*1
        x = current_segment.max(axis=0)
        y = current_segment.max(axis=1)
        x_max = np.max(x*np.arange(len(x)))  # max x point of border
        x[x == 0] = x.shape[0]
        x_min = np.argmin(x)  # min x point of border
        y_max = np.max(y*np.arange(len(y)))  # max y point of border
        y[y == 0] = y.shape[0]
        y_min = np.argmin(y)  # min y point of border

        # If there is enough space, border boxes should be 12 pixels away from the actual segment border
        if x_min-segment_border_in_pixel > 0:
            x_min -= segment_border_in_pixel
        else:
            x_min = 0
        if y_min-segment_border_in_pixel > 0:
            y_min -= segment_border_in_pixel
        else:
            y_min = 0
        
        if x_max+segment_border_in_pixel < width:
            x_max += segment_border_in_pixel
        else:
            x_max = width
        if y_max+segment_border_in_pixel < height:
            y_max += segment_border_in_pixel
        else:
            y_max = height

        # Append the min, max values of borders to the segment metadata list
        segment_meta_data_per_file.append([[x_min, x_max, y_min, y_max]])

        # Append the spectrogram segment to the segment list
        segment_image = specgram_image_processed_for_template_matching[y_min:y_max + 1, x_min:x_max + 1]
        segment_data_per_file.append([[x_min, x_max, y_min, y_max], segment_image])
    
    # Sort segment_meta_data_per_file using x_min as key
    segment_meta_data_per_file = sorted(
        segment_meta_data_per_file, key=lambda segment_location: segment_location[:][0][0])
    segment_data_per_file = sorted(
        segment_data_per_file, key=lambda segment_location: segment_location[:][0][0])

    # Return segment metadata (contains border positions of segments) and segment data per file (spectrogram segments
    # and metadata)
    return segment_meta_data_per_file, segment_data_per_file


def preprocess_specgram_image_for_labeling(specgram_image_cut_):
    """
    Reduce backgound noise by median clipping, binary closing to close small holes,
    dilation to make segments wider, median filtering for extra noise reduction,
    and remove objects that are smaller than 50 connected components
    :param specgram_image_cut_:
    :return:
    """
    s1 = remove_noise_per_freqband_and_time_frame(specgram_image_cut_)
    s2 = binary_closing(s1, structure=np.ones((6, 10)))
    s3 = binary_dilation(s2, np.ones((3, 5)))  # bigger
    s4 = median_filter(s3, size=(5, 3))
    s5 = remove_small_objects(s4, 50)

    # Visualize and save the preprocessed images
    visualize_image_preprocessing([specgram_image_cut_, s1, s3, s5])
    
    return s5


timestamp_start = time.clock()  # for performance measurement
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), "\n")


def process_all_files(file_type_):
    
    global specgram_image_cut
    global segmented_filename_without_ext
    global num_of_segments
    global total_num_of_segments
    global segment_metadata_per_file
    
    print('\nProcessing' + file_type_ + 'Files ...')

    # If save images flag is on, create folders to save spectrogram images
    if SaveImagesOfSegmentedSpecgramsFlag:

        images_of_segmentation_dir = working_directory + file_type_ + 'Data/ImagesOfSegmentation/'
        if not os.path.exists(images_of_segmentation_dir):
            os.makedirs(images_of_segmentation_dir)
    
        images_of_pre_processing_dir = working_directory + file_type_ + 'Data/ImagesOfPreProcessing/'
        if not os.path.exists(images_of_pre_processing_dir):
            os.makedirs(images_of_pre_processing_dir)

    total_num_of_segments = 0
    segment_meta_data = []
    segment_data = []

    # For all wav files
    for ti in range(num_of_files):
        
        segment_meta_data.append([])
        segment_data.append([])

        # For each file, get file name, and create spectrogram image of the file (256*number of frames)
        if file_type_ == 'Train':
            segmented_filename_without_ext = train_file_index_to_train_file_name_without_ext(ti)
            file_name = segmented_filename_without_ext + '.wav'
            specgram_image = create_specgram_image_of_audio_file(audio_files_4_training_dir + file_name)
        
        if file_type_ == 'Test':
            segmented_filename_without_ext = test_file_index_to_test_file_name_without_ext(ti)
            file_name = segmented_filename_without_ext + '.wav'
            specgram_image = create_specgram_image_of_audio_file(audio_files_4_testing_dir + file_name)

        # Get rid of 4 lowest and 24 highest frequency bins, resulting in 228 bins in range 170 Hz - 10000 Hz
        # Image becomes of size 228*number of frames
        specgram_image_cut = \
            specgram_image[num_of_high_freq_bins_skipped:int(FFT_SIZE / 2 - num_of_low_freq_bins_skipped), :]

        # Apply Gaussian blurring with sigma = 1.5 to the spectrogram
        specgram_image_cut_processed_for_template_matching = process_image_for_template_matching(specgram_image_cut)

        # Apply pre-processing (median clipping, binary closing & dilation, median filter and small object removal)
        specgram_image_for_labeling = preprocess_specgram_image_for_labeling(specgram_image_cut)
        
        # Segment only test files or train files associated with one or more sound classes
        if file_type_ == 'Test' or len(list_of_bird_classes_per_train_file[ti]) > 0:

            segment_metadata_per_file, segment_data_per_file = \
                label_specgram(specgram_image_for_labeling, specgram_image_cut_processed_for_template_matching)

            # Create list of lists of segment data and metadata: each index contains data from one sound file
            segment_meta_data[ti] = [segment_metadata_per_file]
            segment_data[ti] = [segment_data_per_file]
            num_of_segments = len(segment_metadata_per_file)
        else:
            segment_meta_data[ti] = [[]]
            segment_data[ti] = [[]]
            num_of_segments = 0

        if SaveImagesOfSegmentedSpecgramsFlag:
            save_images_of_segmented_specgrams()
        
        total_num_of_segments += num_of_segments
        print(segmented_filename_without_ext, ': NumOfSegments:', num_of_segments, '\t-->', total_num_of_segments)

    # Save the segment data and metadata to disk
    output = open(working_directory + file_type_ + 'Data/_SegmentMetaData.pkl', 'wb')
    pickle.dump(segment_meta_data, output)
    output.close()
    
    output = open(working_directory + file_type_ + 'Data/_SegmentData.pkl', 'wb')
    pickle.dump(segment_data, output)
    output.close()

# ==================================  MAIN  ==============================================

# Get metadata from files
num_of_train_files, num_of_bird_classes, list_of_bird_classes_per_train_file, \
    num_of_bird_classes_per_train_file, list_of_train_files_per_bird_class = get_nips4b_metadata()

# globals
specgram_image_cut = 0
segmented_filename_without_ext = ''
num_of_segments = 0
total_num_of_segments = 0
segment_metadata_per_file = 0

file_type = 'Train'
num_of_files = 687
process_all_files(file_type)

file_type = 'Test'
num_of_files = 1000
process_all_files(file_type)

print('\nElapsedTime [s]: ', (time.clock() - timestamp_start))
