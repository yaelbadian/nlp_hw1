import pandas as pd
import numpy as np
from pretraining import Features
from optimization import Optimization
from inference import Viterbi
import pickle
import evaluation


file_path = "data/train1.wtag"
test_path = "data/test1.wtag"
features = Features(file_path)
mat = features.create_features()
list_of_mats = features.create_all_mats()

weights = Optimization.optimize_weights(mat, list_of_mats)
weights_path = 'trained_weights_data_1.pkl'  # i identifies which dataset this is trained on
# print(weights)
with open(weights_path, 'wb') as f:
    pickle.dump(weights, f)

viterbi = Viterbi(features, weights)
list_on_sentences, real_list_of_tags = evaluation.prepare_test_data(test_path)
pred_list_of_tags = viterbi.predict_tags(list_on_sentences)
accuracy, accuracies = evaluation.calculate_accuracy(real_list_of_tags, pred_list_of_tags)
print(accuracy)
print(accuracies)





