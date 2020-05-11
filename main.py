import pandas as pd
import numpy as np
from pretraining import Features
from optimization import Optimization
from inference import Viterbi
import pickle
import evaluation


def print_features_and_weights(weights, features):
    lst = []
    for i, weight in enumerate(weights):
        lst.append((features.idx_to_feature[i], weight))
    lst.sort(key=lambda x: abs(x[1]), reverse=True)
    print(lst[:1000])


file_path = "data/train1.wtag"
test_path = "data/test1.wtag"
# features = Features(file_path)
# mat = features.create_features()
# list_of_mats = features.create_all_mats()
# features.print_statistics('word_ctag', features.word_ctag_count)
# features.print_statistics('prefix', features.prefix_count)
# features.print_statistics('suffix', features.suffix_count)
# features.print_statistics('pptag_ptag_ctag', features.pptag_ptag_ctag_count)
# features.print_statistics('ptag_ctag', features.ptag_ctag_count)
# features.print_statistics('ctag', features.ctag_count)
# features.print_statistics('pword_ctag', features.pword_ctag_count)
# features.print_statistics('nword_ctag', features.nword_ctag_count)
# features.print_statistics('len_word', features.len_word_count)
# features.print_statistics('upper_lower_number', features.upper_lower_number_count)
# features.print_statistics('punctuation_starts', features.punctuation_starts_count)
# features.print_statistics('punctuation', features.punctuation_count)
# features.print_statistics('num_of_uppers', features.num_of_uppers_count)
# features.print_statistics('is_number', features.is_number_count)

# features.save('features_updated.pkl')
features = Features.load('features_updated.pkl')
mat = features.create_features()
list_of_mats = features.create_all_mats()

weights = Optimization.optimize_weights(mat, list_of_mats)
print_features_and_weights(weights, features)
weights_path = 'trained_weights_data_1_updated.pkl'  # i identifies which dataset this is trained on
print(weights)
with open(weights_path, 'wb') as f:
    pickle.dump(weights, f)




# with open(weights_path, 'rb') as file:
#     weights = pickle.load(file)
#
#
# viterbi = Viterbi(features, weights)
# list_on_sentences, real_list_of_tags = evaluation.prepare_test_data(test_path)
# pred_list_of_tags = viterbi.predict_tags(list_on_sentences[:1])
# accuracy, accuracies = evaluation.calculate_accuracy(real_list_of_tags[:1], pred_list_of_tags[:1])
# print(accuracy)
# print(accuracies)





