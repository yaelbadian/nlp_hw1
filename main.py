import pandas as pd
import numpy as np
from pretraining import Features
from optimization import Optimization
from inference import Viterbi
import pickle
import evaluation
from macros import experiments
import random


def print_features_and_weights(weights, features):
    lst = []
    for i, weight in enumerate(weights):
        lst.append((features.idx_to_feature[i], weight))
    lst.sort(key=lambda x: abs(x[1]), reverse=True)
    print(lst[:1000])


def create_features(thresholds, file_path, features_path):
    features = Features(file_path, thresholds)
    features.save(features_path)
    mat = features.create_features()
    list_of_mats = features.create_all_mats()
    return features, mat, list_of_mats


def optimize(mat, list_of_mats, weights_path):
    weights, likelihood = Optimization.optimize_weights(mat, list_of_mats)
    with open(weights_path, 'wb') as f:
        pickle.dump(weights, f)
    return likelihood, weights


def viterbi_test(test_path, features, weights):
    viterbi = Viterbi(features, weights)
    list_of_sentences, real_list_of_tags = evaluation.prepare_test_data(test_path)
    pred_list_of_tags = viterbi.predict_tags(list_of_sentences)
    # for i in range(20):
    #     print("SENTENCE", i)
    #     print(list_on_sentences[i])
    #     for word, t1, t2 in zip(list_on_sentences[i].split(' '), real_list_of_tags[i], pred_list_of_tags[i]):
    #         if t1 != t2:
    #             print(word, t1, t2)
    accuracy, accuracies = evaluation.calculate_accuracy(real_list_of_tags, pred_list_of_tags)
    return accuracy, accuracies


def viterbi_comp(comp_path, features, weights, output_path):
    list_of_sentences = evaluation.prepare_comp_data(comp_path)
    viterbi = Viterbi(features, weights)
    pred_list_of_tags = viterbi.predict_tags(list_of_sentences)
    with open(output_path, "w") as file:
        for i in range(len(list_of_sentences)):
            sentence = ''
            for word, tag in zip(list_of_sentences[i].split(' '), pred_list_of_tags[i]):
                sentence += word + '_' + tag
            file.write(sentence + '\n')


def split_train_test(list_of_sentences, list_of_tags, test_size):
    n = len(list_of_sentences)
    indices = random.sample(list(range(n)), int(n*test_size))
    train_sentences, test_sentences, train_tags, test_tags = [], [], [], []
    for i in range(n):
        if i in indices:
            test_sentences.append(list_of_sentences[i])
            test_tags.append(list_of_tags[i])
        else:
            train_sentences.append(list_of_sentences[i])
            train_tags.append(list_of_tags[i])
    return train_sentences, test_sentences, train_tags, test_tags





list_on_sentences, list_of_tags = evaluation.prepare_test_data(test_path)
train_sentences, test_sentences, train_tags, test_tags = split_train_test(list_of_sentences, list_of_tags, test_size)


if __name__ == '__main__':
    file_path = "data/train1.wtag"
    test_path = "data/test1.wtag"
    for experiment, thresholds in experiments.items():
        features_path = 'experiments/' + experiment + '_features.pkl'
        weights_path = 'experiments/' + experiment + '_weights.pkl'
        features, mat, list_of_mats = create_features(thresholds, file_path, features_path)
        likelihood, weights = optimize(mat, list_of_mats, weights_path)
        accuracy, accuracies = viterbi_test(test_path, features, weights)
        print("results for experiment: {}, likelihood: {}, accuracy: {}".format(experiment, likelihood, accuracy))
        print(accuracies)
        with open('experiments/results.txt', 'a') as file:
            file.write("results for experiment: {}, likelihood: {}, accuracy: {}\n".format(experiment, likelihood, accuracy))

    # with open('experiments/exp_1_features.pkl', 'rb') as file:
    #     features = pickle.load(file)
    #
    # with open('experiments/exp_1_weights.pkl', 'rb') as file:
    #     weights = pickle.load(file)
    #
    # viterbi_alg(test_path, features, weights)
    #









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

