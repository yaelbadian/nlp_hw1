import pandas as pd
import numpy as np
from pretraining import feature_statistics_class
from optimization import Optimization
import inference


file_path = "data/train1.wtag"
features = feature_statistics_class(file_path)
mat = features.create_features()
list_of_mats = features.create_all_mats()

v = np.ones(mat.shape[1])
opt = Optimization(mat, list_of_mats, 0.1)
print(opt.calc_objective_per_iter(v))

