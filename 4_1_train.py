"""
This file is used for training model-O, making predictions on the test set, and writing the predictions to csv files for
submission.

Adapted and modified from Mario Lasseck's implementation
"""

import numpy as np
import pandas as pd
import _pickle as pickle
from sklearn.ensemble import ExtraTreesRegressor

from sklearn import metrics

import glob
import os
import time
import datetime

from GlobalStuff import working_directory, train_labels_dir, audio_files_4_testing_dir, submission_directory
from Util import get_nips4b_metadata


def log(string):
    
    log_fid = working_directory + '_ClassificationResultsLog.txt'
    
    with open(log_fid, 'a') as log_file:
        log_file.write(string + '\n')
    log_file.close()


def get_nips4b_train_data():
    """
    Get the training labels and data required for training
    :return:
    """

    train_labels_fid = train_labels_dir + 'numero_file_train.txt'
    train_segment_files_dir = working_directory + 'TrainData/SegmentProbabilitiesPerFile/'

    # Load the training file labels
    train_data = np.loadtxt(train_labels_fid, dtype=np.int32)

    # Get rid of index row and time row
    # train_data = train_data[:, 1:train_data.shape[1]-1]
    train_data = train_data[:, 1:train_data.shape[1]]

    # Get number of training files and number of bird classes
    # Sum up the columns to find out the number of bird classes per file (SUM INCLUDES DURATION OF FILE AS WELL)
    num_of_train_files_, num_of_classes_ = train_data.shape
    num_of_birds_per_file = np.sum(train_data, axis=1)
    train_data = pd.DataFrame(train_data)
     
    # Add number of bird species per file column
    train_data['num_of_birds_per_file'] = num_of_birds_per_file
     
    # Add file name column
    train_file_names = []
    for i in range(num_of_train_files_):
        train_file_names.append('nips4b_birds_trainfile' + '{0:03d}'.format(i+1) + '.wav')
     
    train_data['FileName'] = train_file_names
     
    # Add has feature vector? Column
    has_feature_vector = np.zeros(num_of_train_files_, dtype=np.int32)

    feature_vectors_missing = 0
    train_data['HasFeatureVec'] = has_feature_vector

    # Get all the segment probability training files and put into a list
    os.chdir(train_segment_files_dir)
    feature_vector_file_names = []
    for feature_vector_file_name in glob.glob('*.pkl'):
        feature_vector_file_names.append(feature_vector_file_name[:-4])

    # Get all train file names and get rid of file extension
    for i in range(num_of_train_files_):
        file_name_without_ext = train_data['FileName'].values[i][:-4]

        # If the current training file has feature vectors (segment probabilities) update the has feature vector column
        # in the train data matrix
        if file_name_without_ext in feature_vector_file_names:
            train_data['HasFeatureVec'].values[i] = 1
        else:
            print(file_name_without_ext, 'missing!')
            feature_vectors_missing = 1
    
    # reduce TrainData if .pkl Files are missing
    if feature_vectors_missing:
        train_data = train_data[train_data.HasFeatureVec == 1]
        num_of_train_files_ = len(train_data.index)
        print('NumOfTrainFilesWithFeatures: ', num_of_train_files_)
     
    train_feature_list = []
    # For each training file
    for i in range(num_of_train_files_):

        # Get file name without extension
        file_name_without_ext = train_data['FileName'].values[i][:-4]

        # Import the features (segment probabilities) for the current file
        pkl_file_ = open(train_segment_files_dir + file_name_without_ext + '.pkl', 'rb')
        feature_vector_of_current_file = pickle.load(pkl_file_)
        pkl_file_.close()

        # Select features (append all segment probabilities of current file to one continuous list
        # of size total number of segments
        selected_features = []
        
        for ti_ in range(num_of_train_files_):
            selected_features = np.concatenate((selected_features, feature_vector_of_current_file[ti_]), axis=0)

        # Append the above list to a training feature list (each item represents a training file
        # and is of length total number of segments)
        train_feature_list.append(selected_features)

    # Convert to numpy array
    train_feature_matrix = np.array(train_feature_list)

    # If include file and segment stats flag is on add file and segment stats to the training matrix
    if include_file_and_segment_properties:
        pkl_file_ = open(working_directory + 'TrainData/_FileAndSegmentStatisticsScaled.pkl', 'rb')
        file_and_segment_properties_train_row_scaled = pickle.load(pkl_file_)
        pkl_file_.close()
        num_of_file_and_segment_properties = file_and_segment_properties_train_row_scaled.shape[1]
        train_feature_matrix = np.hstack((train_feature_matrix, file_and_segment_properties_train_row_scaled))
    else:
        num_of_file_and_segment_properties = 0

    # Print the number of features
    num_of_features = train_feature_matrix.shape[1]
    print('num_of_features: ', num_of_features)

    # Add names f_n as column names of the training matrix where n is the index number of feature (segment prob)
    feature_ids_ = ['F_'+str(x) for x in range(num_of_features)]
    train_feature_matrix__df = pd.DataFrame(train_feature_matrix, columns=feature_ids_)

    # Add file name index column
    train_feature_matrix__df['FileName'] = train_data.index
    
    # Check for NaNs
    nan_indices = pd.isnull(train_feature_matrix__df).any(1).nonzero()[0]
    if len(nan_indices) > 0:
        print('nan_indices:', nan_indices)
        print('NaNFileNames:', train_data['FileName'].values[nan_indices])
        train_feature_matrix__df = train_feature_matrix__df.fillna(0)
         
    # get rid of NaNs
    # train_feature_matrix__df = train_feature_matrix__df.fillNa(0)

    # Merge the train data matrix (contains labels, number of bird species per file, file name, has feature vector?)
    # with the training feature matrix (contains segment probs, file and segment statistics)
    train_data = pd.merge(left=train_data, right=train_feature_matrix__df, left_index=True, right_on='FileName')

    return num_of_train_files_, num_of_classes_, feature_ids_, num_of_file_and_segment_properties, train_data


def get_nips4b_test_data():
    """
    Get the test data information required for classification
    :return:
    """
    test_feature_files_dir = working_directory + 'TestData/SegmentProbabilitiesPerFile/'
    
    if os.path.exists(test_feature_files_dir):
        os.chdir(audio_files_4_testing_dir)
        test_audio_file_names = []

        # Get all test files
        for TestAudioFileName in glob.glob('*.wav'):
            test_audio_file_names.append(TestAudioFileName[:-4])
        
        num_of_test_files_ = len(test_audio_file_names)
        print('num_of_test_files:', num_of_test_files_)

        test_feature_list = []
        feature_vec_file_names = []
        os.chdir(test_feature_files_dir)

        # Get all segment probabilities of test files
        for feature_vector_file_name in glob.glob('*.pkl'):
            feature_vec_file_names.append(feature_vector_file_name[:-4])

        # For each test file
        for i in range(num_of_test_files_):
            file_name_without_ext = test_audio_file_names[i]

            # If segment probabilities for given test file exists, get segment probabilities of current file
            if file_name_without_ext in feature_vec_file_names:
                pkl_file_ = open(test_feature_files_dir + file_name_without_ext + '.pkl', 'rb')
                feature_vec_per_file = pickle.load(pkl_file_)
                pkl_file_.close()

                # Select features (append all segment probabilities of current file to one continuous list
                # of size total number of segments
                selected_features = []
                
                for ti_ in range(num_of_train_files):
                    selected_features = np.concatenate((selected_features, feature_vec_per_file[ti_]), axis=0)

                # Append the above list to a test feature list (each item represents a test file
                # and is of length total number of segments)
                test_feature_list.append(selected_features)

            else:
                print(file_name_without_ext, 'missing FeatureFile!')

        # Convert to numpy array
        test_feature_matrix = np.array(test_feature_list)

        # If include file and segment stats flag is on add file and segment stats to the training matrix
        if include_file_and_segment_properties:
            pkl_file_ = open(working_directory + 'TestData/_FileAndSegmentStatisticsScaled.pkl', 'rb')
            file_and_segment_properties_test_row_scaled = pickle.load(pkl_file_)
            pkl_file_.close()
            test_feature_matrix = np.hstack((test_feature_matrix, file_and_segment_properties_test_row_scaled))

        # Print the number of features
        num_of_features = test_feature_matrix.shape[1]

        # Add names f_n as column names of the test matrix where n is the index number of feature (segment prob)
        feature_ids_ = ['F_'+str(x) for x in range(num_of_features)]
        test_ = pd.DataFrame(test_feature_matrix, columns=feature_ids_)

        # Add file name index column
        test_['FileName'] = test_audio_file_names
        
    else:
        test_ = pd.DataFrame()  # empty DataFrame
        print('no TestFiles')
        
    return test_


def train(num_of_cv_folds_, num_of_estimators_, max_features_, min_sample_split_):

    global train_run_id
    global targets_predicted_per_bird_4_test_files
    global prediction_matrix_test

    # for performance measurement
    time_stamp_start = time.clock()
    date_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    date_time_id = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
    print(date_time, "\n")
    
    print('\nTrainRunId:', train_run_id)
    print('Cvs, MaxNumEstimators, MaxFeaturesForSplit, MinSampleSplit: ')
    print(num_of_cv_folds_, num_of_estimators_, max_features_, min_sample_split_)

    # If eval prediction flag is on
    if eval_prediction:
        targets_true_per_bird4_train_files_matrix = np.zeros((num_of_train_files, num_of_classes))
        targets_predicted_per_bird4_train_files_matrix = np.zeros((num_of_train_files, num_of_classes))
    
    print('Start Training with', num_of_cv_folds_, 'Fold CV ...')

    # This list is of length train files, and each item has a value [0,11]. The value indicates the
    # cross-validation iteration in which the file will act as part of the validation set.
    # For ex: if cv[0] = 6, this means that the 0th training file will be part of the validation set in
    # the 6th cross-validation. There is a total of 12 cross-validation iterations
    cv = np.random.randint(0, num_of_cv_folds_, len(train_.index))
      
    train_['cv'] = cv
    targets_true = []
    targets_predicted = []

    # If the test data feature matrix exists, create a matrix of size 88*1000 to store the the test set probabilities
    if len(test) > 0:
        targets_predicted_per_bird_4_test_files = []
        prediction_matrix_test = np.empty((num_of_cv_folds_, num_of_classes, len(test)), dtype=np.float64)
        for bird_ in range(num_of_classes):
            targets_predicted_per_bird_4_test_files.append(np.zeros(len(test)))

    # For each cross-validation iteration
    for c in range(num_of_cv_folds_):
        current_targets_true = []
        current_targets_predicted = []

        # Get the training and validation set samples for this iteration
        training_set = train_[train_.cv != c]
        validation_set = train_[train_.cv == c]
        validation_set_indices = validation_set.index.tolist()

        # For each bird class
        for bird_ in range(num_of_classes):

            # Get training and validation set samples for the current bird class
            training_set_targets = training_set[bird_]
            validation_set_targets = validation_set[bird_]

            # Get all the segment probability indices associated with the current bird class
            segment_probability_ids_associated_with_current_bird_class = \
                ['F_' + str(x) for x in feature_ids_associated_with_bird_classes[bird_]]

            # If file and segment probability flag is on
            if include_file_and_segment_properties:

                # Get the file and segment probabilities associated with current bird class
                file_and_segment_properties_feature_ids = \
                    ['F_' + str(x) for x in range(len(feature_ids) - num_of_file_and_segment_probs, len(feature_ids))]

                # Combine segment probabilities and file and segment properties
                current_feature_ids = \
                    segment_probability_ids_associated_with_current_bird_class + file_and_segment_properties_feature_ids
            else:
                current_feature_ids = segment_probability_ids_associated_with_current_bird_class

            # Using the feature ids from above, get features of each training and validation set sample
            training_set_features = training_set[current_feature_ids]
            validation_set_features = validation_set[current_feature_ids]

            # Get features of test set using the feature ids from above
            if len(test) > 0:
                test_set_features = test[current_feature_ids]

            # Calculate the number of estimators for the current bird class:
            # Chosen as twice the number of selected features, but not higher than 500
            num_of_estimators_current_bird = int(2*len(current_feature_ids))

            if num_of_estimators_current_bird > num_of_estimators_:
                num_of_estimators_current_bird = num_of_estimators_

            # Build and fit the extra trees classifier
            classifier = ExtraTreesRegressor(n_estimators=num_of_estimators_current_bird, max_features=max_features_,
                                             min_samples_split=min_sample_split_,
                                             random_state=(train_run_id + bird_) * 100, verbose=0)
            classifier.fit(training_set_features, training_set_targets)

            # Make predictions on the validation and test sets
            validation_set_predictions = classifier.predict(validation_set_features)

            if len(test) > 0:
                test_set_predictions = classifier.predict(test_set_features)

            # Convert the pandas dataframes containing true labels and predicted labels into lists
            # Append to list at every bird class
            current_targets_true += list(validation_set_targets)
            current_targets_predicted += list(validation_set_predictions)

            # If evaluate predictions flag is on
            if eval_prediction:

                #
                true_list = list(validation_set_targets)
                predictions_list = list(validation_set_predictions)
                counter = 0
                for ti_ in validation_set_indices:
                    targets_true_per_bird4_train_files_matrix[ti_][bird_] = true_list[counter]
                    targets_predicted_per_bird4_train_files_matrix[ti_][bird_] = predictions_list[counter]
                    counter += 1

            # Add the test set predictions into the the target prediction array
            # Here, the array is of size 88*1000: For bird class zero, we fill the array from [0][0] to [0][999]
            # We divide the numbers by number of cv folds since we want to take the mean from all cross-validation
            # iterations
            # Prediction Matrix Test array also includes results from each cross validation
            if len(test) > 0:
                targets_predicted_per_bird_4_test_files[bird_] += test_set_predictions / num_of_cv_folds_
                prediction_matrix_test[c][bird_][:] = np.array(test_set_predictions)

        # Interim results
        cur_false_positive_rate, cur_true_positive_rate, cur_thresholds = \
            metrics.roc_curve(current_targets_true, current_targets_predicted, pos_label=1)
        
        targets_true = targets_true + list(current_targets_true)
        targets_predicted = targets_predicted + list(current_targets_predicted)
        false_positive_rate, true_positive_rate, thresholds = \
            metrics.roc_curve(targets_true, targets_predicted, pos_label=1)
        
        # Console Out
        print('c' + str(c).zfill(2) + ':', int(metrics.auc(cur_false_positive_rate, cur_true_positive_rate)*10000.0),
              '>', int(metrics.auc(false_positive_rate, true_positive_rate)*10000.0))
    
    false_positive_rate, true_positive_rate, thresholds = \
        metrics.roc_curve(targets_true, targets_predicted, pos_label=1)

    area_under_curve = metrics.auc(false_positive_rate, true_positive_rate)
    
    log('')
    log('')
    log('------------------  ' + date_time + '  ------------------')
    log('')
    log('NumOfTrainFiles:                   ' + str(num_of_train_files))
    log('NumOfFeatures:                     ' + str(len(feature_ids)))
    log('NumOfClasses:                      ' + str(num_of_classes))
    log('')
    log('NumOfCvFolds:                      ' + str(num_of_cv_folds_))
    log('MaxNumOfEstimators:                ' + str(num_of_estimators_))
    log('MaxFeaturesForSplit:               ' + str(max_features_))
    log('MinSampleSplit:                    ' + str(min_sample_split_))
    log('')
    log('area_under_curve:  ' + str(area_under_curve))
    
    print('\narea_under_curve:', area_under_curve)

    write_submission_csv_file(date_time_id, area_under_curve)

    if eval_prediction:
        output = open(working_directory + 'TargetsPredictedPerBird4TrainFilesMatrix_'
                      + str(train_run_id) + '.pkl', 'wb')
        pickle.dump(targets_predicted_per_bird4_train_files_matrix, output)
        output.close()
        
        output = open(working_directory + 'TargetsTruePerBird4TrainFilesMatrix_' + str(train_run_id) + '.pkl', 'wb')
        pickle.dump(targets_true_per_bird4_train_files_matrix, output)
        output.close()
    
    train_run_id += 1
    
    print('\nElapsedTime [s]: ', (time.clock() - time_stamp_start))


def write_submission_csv_file(date_time_id, area_under_curve):
    
    if len(test) > 0:
        score = '-' + str(int(area_under_curve * 10000.0))
        
        submission_file_name = date_time_id + score + '-SubmissionCsvFile.csv'

        submission_id = []
        submission_probability = []
        submission_probability_tuned = []
        
        for f in range(len(test)):
            for c in range(num_of_classes):
                submission_id.append(test['FileName'].values[f] + '.wav_classnumber_' + str(c + 1))
                submission_probability.append(targets_predicted_per_bird_4_test_files[c][f])
                
                # get mean of cv's, discard min and max values 
                prediction_per_cv = np.empty(num_of_cv_folds, dtype=np.float64)
                for cv in range(num_of_cv_folds):
                    prediction_per_cv[cv] = prediction_matrix_test[cv][c][f]
                prediction_per_cv_sorted = np.sort(prediction_per_cv)
                submission_probability_tuned.append(
                    np.mean(prediction_per_cv_sorted[1:num_of_cv_folds - 1]))  # 2 lowest & highest value discarded
        
        submission_df_tuned = pd.DataFrame(submission_id, columns=['ID'])
        submission_df_tuned['Probability'] = submission_probability_tuned
        submission_df_tuned.to_csv(submission_directory + submission_file_name, float_format='%.8f', index=False)
        
    else:
        print('No Testdata --> No SubmissionFile written!')

# ======================= MAIN ==============================
print('Importing Data ...')

# Get training set metadata
num_of_train_files, num_of_bird_classes, list_of_bird_classes_per_train_file, \
    num_of_bird_classes_per_train_file, list_of_train_files_per_bird_class = get_nips4b_metadata()

# Option variables
eval_prediction = 1
# eval_prediction = 1
include_file_and_segment_properties = 1
train_run_id = 0

targets_predicted_per_bird_4_test_files = []  # global
targets_predicted_per_bird_4_test_files_tuned = []  # global

segment_offsets_per_train_file = []
num_of_segments_per_train_file = []

# Load the first segment probability training file
pkl_file = open(working_directory + 'TrainData/SegmentProbabilitiesPerFile/nips4b_birds_trainfile001.pkl', 'rb')
segment_vectors_train_temp = pickle.load(pkl_file)
pkl_file.close()
current_segment_offset = 0

# Iterate through the segment probabilities and create lists that have contain the starting offset
# of segments for each training file
# feature_offset_for_train_index -> each item represents the start index number of segments for each file
# num_of_features_for_train_index -> number of segments in each training file
for ti in range(len(segment_vectors_train_temp)):
    segment_offsets_per_train_file.append(current_segment_offset)
    num_of_segments_per_train_file.append(len(segment_vectors_train_temp[ti]))
    current_segment_offset += len(segment_vectors_train_temp[ti])


# Get number of training files, number of classes, feature IDs, number of file and segment properties, and
# training matrix that contains labels, features, etc
num_of_train_files, num_of_classes, feature_ids, num_of_file_and_segment_probs, train_ = get_nips4b_train_data()


# For each bird class
feature_ids_associated_with_bird_classes = []
for bird in range(num_of_classes):

    # For each training file associated with the current bird class
    feature_ids_associated_with_bird_classes.append([])
    temp_list = []  # Will contain all segment indices associated with given bird class
    for ti in list_of_train_files_per_bird_class[bird]:

        # Get the segment indices of segments that belong to the current training file
        np_array = np.arange(segment_offsets_per_train_file[ti],
                             segment_offsets_per_train_file[ti] + num_of_segments_per_train_file[ti], dtype=np.int32)

        # Convert the array (that contains segment indices of segments) into list and append to the cumulative list
        temp_list += list(np_array)

    # Add the list that contains segment indices associated with given bird class to the list, using
    # the bird class as index (NOTE: CLASS 88 is there by mistake; it is based on the wav file duration column,
    # thus contains all the segments from all the files
    feature_ids_associated_with_bird_classes[bird] = temp_list

    print('NumOfFeaturesForSoundClass', bird, ':',
          len(feature_ids_associated_with_bird_classes[bird]), '+', num_of_file_and_segment_probs)

print('NumOfTrainFiles:', num_of_train_files, '  NumOfClasses:', num_of_classes, 'NumOfFeatures:', len(feature_ids))


# Get the test file features
num_of_test_files = 1000
test = pd.DataFrame()
test = get_nips4b_test_data()

# Parameters to optimize
num_of_cv_folds_list = [12, 14]
num_of_estimators_list = [500]  # Number of trees in the forest
min_sample_split_list = [4, 3]  # Minimum number of samples required to split an internal node
max_features_list = [5, 4]  # Number of features to consider when looking for the best split


for num_of_cv_folds in num_of_cv_folds_list:
        for num_of_estimators in num_of_estimators_list:
            for max_features in max_features_list:
                for min_sample_split in min_sample_split_list:
                    train(num_of_cv_folds, num_of_estimators, max_features, min_sample_split)
