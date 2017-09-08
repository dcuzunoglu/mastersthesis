"""
This file contains the helper functions used throughout the scripts.

Adapted and modified from Mario Lasseck's implementation
"""

import numpy as np
import wave

import matplotlib.pyplot as plt
from scikits.samplerate import resample
from scipy.ndimage.filters import gaussian_filter

from GlobalStuff import FS, FFT_SIZE, FFT_HOP_SIZE, train_labels_dir


def create_specgram_image_of_audio_file(fid):
    
    # Read wav file
    wave_obj = wave.open(fid, 'r')

    # Vectorize the wave file
    sample_vec = np.fromstring(wave_obj.readframes(wave_obj.getnframes()), np.short)/32780.0
    sample_rate = wave_obj.getframerate()  # get sample rate
    wave_obj.close()

    # Re-sample the vector to 22050 Hz
    if not sample_rate == 22050:
        sample_vec = resample(sample_vec, FS/sample_rate, 'sinc_best')
    
    num_of_samples = len(sample_vec)
    
    # Normalize the vector
    sample_vec = 0.9*sample_vec/sample_vec.max()

    # Create spectrogram
    # the window size (FFT size) is 512, and FFT HOP size is 128 -> 128/512 = 0.25 meaning a 75% overlap
    num_of_specgram_frames = int((num_of_samples-FFT_SIZE)/FFT_HOP_SIZE) + 1

    # Hanning window function
    window = 0.5 * (1.0 - np.cos(np.array(range(FFT_SIZE))*2.0*np.pi/(FFT_SIZE-1)))

    # Apply STFT
    specgram = []
    for j in range(num_of_specgram_frames):
        vec = sample_vec[j*FFT_HOP_SIZE: j*FFT_HOP_SIZE + FFT_SIZE] * window
        real_fft = np.fft.rfft(vec, FFT_SIZE)
        real_fft = abs(real_fft[:int(FFT_SIZE/2)])
        specgram.append(real_fft)

    # Create spectrogram image
    specgram_transposed = np.transpose(np.array(specgram))
    specgram_image = np.zeros_like(specgram_transposed)

    for i in range(specgram_transposed.shape[0]):
        specgram_image[i] = specgram_transposed[-i - 1]
    
    if np.max(specgram_image) <= 0.0:
        print('Problem: np.max(specgram_image) <= 0.0')
    else:
        specgram_image /= np.max(specgram_image)
    
    return specgram_image


def create_specgram_image_of_unprocessed_audio_file(fid):
    
    # Read wav file
    wave_obj = wave.open(fid, 'r')

    # Vectorize the wave file
    sample_vec = np.fromstring(wave_obj.readframes(wave_obj.getnframes()), np.short)/32780.0
    sample_rate = wave_obj.getframerate()
    wave_obj.close()

    # Re-sample the vector to 22050 Hz
    if not sample_rate == 22050:
        sample_vec = resample(sample_vec, FS/sample_rate, 'sinc_best')
    
    num_of_samples = len(sample_vec)
    
    # Normalize the vector
    sample_vec = 0.9*sample_vec/sample_vec.max()

    # Create spectrogram
    # the window size (FFT size) is 512, and FFT HOP size is 128 -> 128/512 = 0.25 meaning a 75% overlap
    num_of_specgram_frames = int((num_of_samples - FFT_SIZE) / FFT_HOP_SIZE) + 1

    # Hanning window function
    window = 0.5 * (1.0 - np.cos(np.array(range(FFT_SIZE))*2.0*np.pi/(FFT_SIZE-1)))

    # Apply STFT
    specgram = []
    for j in range(num_of_specgram_frames):
        vec = sample_vec[j * FFT_HOP_SIZE: j * FFT_HOP_SIZE + FFT_SIZE] * window
        real_fft = np.fft.rfft(vec, FFT_SIZE)
        real_fft = abs(real_fft[:int(FFT_SIZE / 2)])
        specgram.append(real_fft)

    # Create spectrogram image
    specgram_transposed = np.transpose(np.array(specgram))
    specgram_image = np.zeros_like(specgram_transposed)

    for i in range(specgram_transposed.shape[0]):
        specgram_image[i] = specgram_transposed[-i - 1]
    
    return specgram_image


def process_image_for_template_matching(image):
    """
    Apply Gaussian blurring to the image
    :param image:
    :return:
    """
    image_processed = gaussian_filter(image, sigma=1.5)  # @UndefinedVariable
    return image_processed


def get_rgb_value(cmap_name, index, range_):
    # cmap_name: 'jet', 'hot_r', 'gray', ...
    # Range: Quantisierung (z.B. 10 --> 10 RgbValues 0...Range-1 
    # Index: 0...Range-1 (z.B. Range-1 --> last RgbValue of cmap
    cmap = plt.cm.get_cmap(cmap_name, range_)
    rgb_arr = cmap(np.arange(range_))[:, :-1]
    return rgb_arr[index]


def pixel2samples(pixel):
    return pixel * FFT_HOP_SIZE


def pixel2seconds(pixel):
    return pixel * FFT_HOP_SIZE / FS


def seconds2samples(seconds):
    return int(seconds*FS)


def seconds2pixel(seconds):
    return int(seconds*FS/FFT_HOP_SIZE)


def freq2pixel(freq_in_hz):
    return int(freq_in_hz * FFT_SIZE / FS)


def pixel2freq(pixel):
    return pixel*FS/FFT_SIZE


def train_file_index_to_train_file_name_without_ext(train_file_index):
    """
    Generates file name given the file index
    :param train_file_index:
    :return:
    """
    train_file_name_without_ext = 'nips4b_birds_trainfile' + '{0:03d}'.format(train_file_index + 1)
    return train_file_name_without_ext


def test_file_index_to_test_file_name_without_ext(test_file_index):
    """
    Generates file name given the file index
    :param test_file_index:
    :return:
    """
    test_file_name_without_ext = 'nips4b_birds_testfile' + '{0:04d}'.format(test_file_index + 1)
    return test_file_name_without_ext


def get_nips4b_metadata():
    """
    Returns the metadata associated with the files:

    :return:
    number of train files, number of bird classes, bird classes that exist in each train file,
    total number of bird classes per file, file indices that contain each bird class
    """
    # Load training labels
    train_labels = np.loadtxt(train_labels_dir + 'numero_file_train.txt', dtype=np.int32)
    
    # Get rid of index row
    train_labels = train_labels[:, 1:train_labels.shape[1]]
    num_of_train_files, num_of_bird_classes = train_labels.shape

    # Sum up the columns to find out the number of bird classes per file (SUM INCLUDES DURATION OF FILE AS WELL)
    num_of_bird_classes_in_current_file = np.sum(train_labels, axis=1)
    num_of_bird_classes_per_train_file = np.where(num_of_bird_classes_in_current_file > 0)[0]

    list_of_train_files_per_bird_class = []

    # For each bird class, find out which files contain that class
    for bird in range(num_of_bird_classes):
        train_file_index_associated_with_current_bird_class = np.where(train_labels[:, bird] > 0)[0]
        list_of_train_files_per_bird_class.append(train_file_index_associated_with_current_bird_class)

    list_of_bird_classes_per_train_file = []

    # For each train file, find out which classes of bird exist in that train file
    for f in range(num_of_train_files):
        bird_classes_in_current_train_file = np.where(train_labels[f, :] > 0)[0]
        list_of_bird_classes_per_train_file.append(bird_classes_in_current_train_file)
    
    return num_of_train_files, num_of_bird_classes, list_of_bird_classes_per_train_file, \
        num_of_bird_classes_per_train_file, list_of_train_files_per_bird_class
