# variable_augmentation_thesis

*******************
This readme contains the instructions to run the scripts I've used in my master's thesis:

Design Matrix Augmentation With Random Variables: An Experiment With the NIPS4b Dataset

The thesis is available on request.
*******************


The dataset for the NIPS4b Bird Challenge can be downloaded from:

https://www.kaggle.com/c/multilabel-bird-species-classification-nips2013

The following libraries for Python 3 were used to run the scripts:

- numpy 1.12.0
- pandas 0.19.2
- scipy 0.18.1
- _pickle
- wave
- matplotlib
- scikit-image-0.13.0
- scikit-learn-0.18.1
- scikits.samplerate-0.4.0.dev
- opencv-python-3.2.0

Before starting to run any of the scripts, the NIPS4b dataset should be downloaded, and the following variables should be changed accordingly from GlobalStuff.py:

- train_labels_dir
- audio_files_4_training_dir
- audio_files_4_testing_dir
- working_directory
--------------------------------------------------------



--------------------------------------------------------
REPRODUCING THE BASELINE MODEL (MODEL-O)

In order to reproduce model-O (a reproduction of Mario Lasseck's approach to NIPS4b), the following scripts can be run in the given order:

- 1_1_segmentAllFiles.py (to segment the spectrograms)

- 2_1_getSegmentProbabilitiesPerTrainFile.py (to create segment probabilities from the segmented spectrograms for training recordings)

- 2_2_getSegmentProbabilitiesPerTestFile.py (to create segment probabilities from the segmented spectrograms for testing recordings)

- 3_1_getFileAndSegmentStatistics.py (to create file and segment statistics features from training and testing recordings)

- 4_1_train.py	(for training, and making predictions on testing files. predictions are also written to csv's for submission)
---------------------------------------------------------



----------------------------------------------------------
REPRODUCING THE MODEL WITH RANDOM VARIABLE AUGMENTATION (MODEL-A)

In order to reproduce model-A, the following script should be run after running 1_1, 2_1, 2_2, and 3_1:

- 4_1_train_data_aug.py

This script is set to be run with fixed validation indices (saved in the outputs/variables directory, these are the same fixed indices we used in our experiments) in order to eliminate the randomization effect arising from using different validation indices for the validation set at each run. If you would like to make a trial run with newly generated validation indices, turn off the "used_fixed_cv_indices" flag. 

In order to save time in the following runs, this model also saves some training and test design matrices, feature ids list, features ids per bird class, and list of train files per bird class to disk after a run. In order to load the saved variables in the following runs, turn on the "load_training_data" and "load_testing_data" flags.

---------------------------------------------------------



---------------------------------------------------------
REPRODUCING THE MODEL WITH FEATURE SELECTION (MODEL-FS)

In order to reproduce model-A, the following scripts should be run after running 1_1, 2_1, 2_2, and 3_1:

- 4_1_train_save_feature_importance.py

This script runs the baseline model and saves the feature importances returned from training to csv files found in the "feature_importances" directory. If this is run multiple_times, the feature importances from multiple runs get added to the csv files, thus the importances from multiple runs can be averaged later on.

- feature_importance_ranking.py

This script averages the feature importance rankings returned from the runs of 4_1_train_save_feature_importance.py, and sorts them. Then it writes the ranked and sorted features into csv files.

- 4_1_train_fsda.py

This script runs the model with feature selection by using the ranked feature csv files returned from feature_importance_ranking.py. By default, this script is set to be run with newly generated validation indices each time, but if you would like to run the fixed cv indices, turn on the "use_fixed_cv_indices" flag.

In order to save time in the following runs, this model also saves some training and test design matrices, feature ids list, features ids per bird class, and list of train files per bird class to disk after a run. In order to load the saved variables in the following runs, turn on the "load_training_data" and "load_testing_data" flags.

--------------------------------------------------------



--------------------------------------------------------
REPRODUCING THE MODEL WITH FEATURE SELECTION AND RANDOM VARIABLE AUGMENTATION (MODEL-FS-A)

In order to reproduce model-FS-A, the following scripts should be run after running 1_1, 2_1, 2_2, and 3_1:

- 4_1_train_save_feature_importance.py

This script runs the baseline model and saves the feature importances returned from training to csv files found in the "feature_importances" directory. If this is run multiple_times, the feature importances from multiple runs get added to the csv files, thus the importances from multiple runs can be averaged later on.

- feature_importance_ranking.py

This script averages the feature importance rankings returned from the runs of 4_1_train_save_feature_importance.py, and sorts them. Then it writes the ranked and sorted features into csv files.

*********************************************
NOTE: If feature importances are already saved and sorted from reproducing model-FS, there is no need to run 4_1_train_save_feature_importance.py and feature_importance_ranking.py again.
********************************************

- 4_1_train_fsda.py

This is the same script used to run model-FS, but some changes need to be done first before reproducing model-FS-A with this. First, "use_fixed_cv_indices" flag on line 697 needs to be turned on. This is because we need to eliminate the effect of randomizing the validation set when we are making runs with variable augmentation. 


Then, line 711 needs to be commented out and line 710 that contains the variable augmentation levels should be uncommented. This would enable the script to use variable augmentation after feature selection.

In order to save time in the following runs, this model also saves some training and test design matrices, feature ids list, features ids per bird class, and list of train files per bird class to disk after a run. In order to load the saved variables in the following runs, turn on the "load_training_data" and "load_testing_data" flags.

-------------------------------------------------------



-------------------------------------------------------
CALCULATING VALIDATION AUC PER CLASS

In order to calculate the validation AUC per class after training a model, the "AUC_per_class.py" script is used. This script saves the validation AUC of each class to a csv.

In the script, the "predictions" variable needs to be pointed at the corresponding .pkl file of the form "TargetsPredictedPerBird4TrainFilesMatrix_xx.pkl" and the true_lables variable needs to be pointed at the corresponding "TargetsTruePerBird4TrainFilesMatrix_xx.pkl" where "xx" refers to the training ID number. The name of the csv file produced by this script can be changed from line 30 to differentiate between the results from different files.

--------------------------------------------------------



---------------------------------------------------------
SUBMITTING PREDICTIONS FOR EVALUATION ON KAGGLE

Our predictions contain 88 rows per test file instead of 87 rows per test file (e.g. one row per bird class). The additional row per file is caused by the file duration column in the training and testing labels in addition to the class columns. Thus, in order to submit for evaluations, the class 88 rows need to be removed from the prediction csv's. To do this, we use the "remove_88.py" script. In the script, simply point the input_csv variable to the csv submission file you want to fix. The script will then produce a submission script in the same directory as the input script with class 88 removed, which can be submitted for evaluation.

---------------------------------------------------------




---------------------------------------------------------
EXPERIMENTAL RESULTS

The directory "experimental_results" contains our feature rankings produced from our experiments. In addition, the directory contains the results from each individual trial run from our experiments.




