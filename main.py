import pandas as pd
import numpy as np
from pretraining import feature_statistics_class
from optimization import Optimization
import inference
import pickle


file_path = "data/train1.wtag"
features = feature_statistics_class(file_path)
mat = features.create_features()
list_of_mats = features.create_all_mats()

weights = Optimization.optimize_weights(mat, list_of_mats)
weights_path = 'trained_weights_data_1.pkl'  # i identifies which dataset this is trained on
print(weights)
with open(weights_path, 'wb') as f:
    pickle.dump(weights, f)




