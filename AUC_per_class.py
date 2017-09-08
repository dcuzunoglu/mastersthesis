"""
This script calculates the validation AUC per class for a training run.

The predictions variable, and the true_labels variable need to be pointed to the according .pkl files. The final
output file name should be changed after each run to differentiate between outputs from this script as well.

Consult readme.txt for more details.
"""

import os
import _pickle as pickle
from sklearn import metrics
import numpy as np

# Main ===================
current_path = os.path.dirname(os.path.realpath(__file__))
validation_results_dir = current_path + '/validation_results/'

if not os.path.exists(validation_results_dir):
    os.makedirs(validation_results_dir)

pkl_file_ = open(validation_results_dir + 'TargetsPredictedPerBird4TrainFilesMatrix_21.pkl', 'rb')
predictions = pickle.load(pkl_file_)
predictions = predictions[:, :87]
pkl_file_.close()
pkl_file_ = open(validation_results_dir + 'TargetsTruePerBird4TrainFilesMatrix_21.pkl', 'rb')
true_labels = pickle.load(pkl_file_)
true_labels = true_labels[:, :87]
pkl_file_.close()
auc_per_class = np.zeros((87, 1))

for i in range(87):

    false_positive_rate, true_positive_rate, thresholds = \
        metrics.roc_curve(true_labels[:, i], predictions[:, i], pos_label=1)

    area_under_curve = metrics.auc(false_positive_rate, true_positive_rate)
    auc_per_class[i] = area_under_curve
np.savetxt("model-FS-A_trial3_AUC_per_Class.csv", auc_per_class, delimiter=",")