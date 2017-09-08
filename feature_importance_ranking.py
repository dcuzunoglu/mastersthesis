"""
This file averages and sorts the feature importance values to be used with our models.

"""

import os
import pandas as pd
import _pickle as pickle

# Main ===================
current_path = os.path.dirname(os.path.realpath(__file__))
feature_importance_dir = current_path + '/feature_importances/'

for i in range(88):
    fi_ranked = pd.read_csv(feature_importance_dir + "class_" + str(i) + ".csv")
    fi_ranked = fi_ranked.sum(axis=0)
    fi_ranked = fi_ranked.sort_values(ascending=False)
    fi_ranked = pd.DataFrame({'Feature Number': fi_ranked.index, 'Importance': fi_ranked.values})
    output = open(feature_importance_dir + "class_" + str(i) + ".pkl", 'wb')
    pickle.dump(fi_ranked, output)
    output.close()
    fi_ranked.to_csv(feature_importance_dir + "class_" + str(i) + "_sorted.csv")

