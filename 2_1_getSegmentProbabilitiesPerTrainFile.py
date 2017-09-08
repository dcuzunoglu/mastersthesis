"""
This file is used to create segment probability features from segmented spectrograms for testing recordings

Adapted and modified from Mario Lasseck's implementation
"""

import numpy as np
import time
import datetime
import _pickle as pickle
import glob
import os
import cv2

from GlobalStuff import num_of_low_freq_bins_skipped, \
    num_of_high_freq_bins_skipped, FFT_SIZE, working_directory, audio_files_4_training_dir
from Util import create_specgram_image_of_audio_file, process_image_for_template_matching

# ====================== MAIN ========================================
audio_files_directory = audio_files_4_training_dir
segment_probabilities_per_file_directory = working_directory + 'TrainData/SegmentProbabilitiesPerFile/'

# Create a folder to save the segment probabilities in
if not os.path.exists(segment_probabilities_per_file_directory):
    os.makedirs(segment_probabilities_per_file_directory)

segment_data_fid = working_directory + 'TrainData/_SegmentData.pkl'

# For performance measurement
timestamp_start = time.clock()
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), "\n")

# Import segments from pkl files
pkl_file = open(segment_data_fid, 'rb')
segment_data = pickle.load(pkl_file)
pkl_file.close()

num_of_train_files = len(segment_data)
print("NumOfTrainFiles: ", num_of_train_files)

target_file_names = []
 
# Create a list that contains names of all files in the train file directory
os.chdir(audio_files_directory)
for target_file_name in glob.glob('*.wav'):
    target_file_names.append(target_file_name)

# Best matches will be searched within the frequency range of each segment with a tolerance of +-4 pixels
freq_tolerance = 4
nans_detected = False

# For each file in the training file list
f = 0
while f < len(target_file_names):

    # Remove the file name extension and add .pkl extension to save segment probabilities
    target_file_name = target_file_names[f]
    file_name_without_extension = target_file_name[:-4]
    segment_probabilities_per_file_fid = segment_probabilities_per_file_directory + file_name_without_extension + '.pkl'
    
    # template_matching_results_per_target_file = []  # List of Length: NumOfTrainFiles
     
    if not os.path.isfile(segment_probabilities_per_file_fid):
         
        # Save the segment probability file without any info (as a dummy)
        output = open(segment_probabilities_per_file_fid, 'wb')
        pickle.dump(0, output)
        output.close()

        print(f, ': ', file_name_without_extension)

        # Create spectrogram image of the current audio file
        specgram_image = create_specgram_image_of_audio_file(audio_files_directory + target_file_name)

        # Get rid of 4 lowest and 24 highest frequency bins, resulting in 228 bins in range 170 Hz - 10000 Hz
        # Image becomes of size 228*number of frames
        specgram_image_cut = \
            specgram_image[num_of_high_freq_bins_skipped:int(FFT_SIZE / 2 - num_of_low_freq_bins_skipped), :]

        # Apply Gaussian blurring with sigma = 1.5 to the spectrogram (this now becomes the target image)
        specgram_image_cut_processed_for_template_matching = process_image_for_template_matching(specgram_image_cut)
        target_specgram_image = specgram_image_cut_processed_for_template_matching

        # Get width of target image
        target_image_width = target_specgram_image.shape[1]

        # For each training file
        segment_probabilities_per_target_file = []
        for ti in range(num_of_train_files):

            # Create a list of lists (the lists inside contain info about each file)
            segment_probabilities_per_target_file.append([])

            # Get segment information about the current file
            segment_data_of_current_file = segment_data[ti][0]
            num_of_segments_of_current_file = len(segment_data_of_current_file)
            
            probabilities = []
            position_x = []
            position_y = []

            # For each segment in the current file
            for s in range(num_of_segments_of_current_file):

                # Get the border pixel positions of segment, calculate width and height
                [x_min, x_max, y_min, y_max] = segment_data_of_current_file[s][0]
                template_width = x_max - x_min
                template_height = y_max - y_min

                # Template width has to be smaller than the target image width
                if template_width < target_image_width:

                    # Get the current segment's spectrogram
                    template_processed = segment_data_of_current_file[s][1]

                    # If the lower y-axis border pixel is greater than the frequency tolerance,
                    # the lower y-axis target for matching equals to (y min- tolerance).
                    # Else, lower y-axis target is zero
                    if y_min > freq_tolerance:
                        y_min_target = y_min - freq_tolerance
                    else:
                        y_min_target = 0

                    # If higher y-axis border is smaller than 247,
                    # the higher y-axis target for matching equals to (y max + tolerance)
                    # Else higher y-axis target is 251
                    if y_max < 0.5*FFT_SIZE-num_of_low_freq_bins_skipped-1-freq_tolerance:
                        y_max_target = y_max + freq_tolerance
                    else:
                        y_max_target = 0.5*FFT_SIZE - num_of_low_freq_bins_skipped - 1

                    # Cut the target spectrogram image from the positions of y-max and y-min targets for
                    # template matching (width of image is same, but height is now smaller)
                    target_specgram_image_reduced_to_freq_5 = target_specgram_image[y_min_target:y_max_target + 1, :]
                     
                    # Use the TM_CCOEFF_NORMED template matching method to
                    # match the segment with the target image
                    # This returns a convoluted image of size (W-w+1, H-h+1) with each position denoting the
                    # probability of template image matching a position on the target image
                    template_matching_method = 5
                    result = cv2.matchTemplate(target_specgram_image_reduced_to_freq_5.astype(np.float32),
                                               template_processed.astype(np.float32), template_matching_method)

                    # Get the positions and values of highest probability matches
                    min_result, max_result, min_loc, max_location = cv2.minMaxLoc(result)

                    # Find the positions on the original target image
                    max_location_x = max_location[0]
                    max_location_y = max_location[1] + y_min_target
                          
                else:
                    max_result = 0.0
                    max_location_x = 0
                    max_location_y = 0

                # Append the maximum probability and its location to the lists accordingly
                probabilities.append(max_result)
                position_x.append(max_location_x)
                position_y.append(max_location_y)

            # Append each file's segment probability and position info to the template matching results list
            # template_matching_results_per_target_file.append([probabilities, position_x, position_y])
            
            if len(probabilities) > 0:
                segment_probabilities_per_target_file[ti] = probabilities

        # Save the template matching probabilities of each file to disk
        output = open(segment_probabilities_per_file_fid, 'wb')
        pickle.dump(segment_probabilities_per_target_file, output)
        output.close()
         
        f = 0
    else:
        f += 1

print("\nElapsedTime [s]: ", (time.clock() - timestamp_start))
