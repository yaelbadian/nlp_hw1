from pretraining import Features
from optimization import Optimization
from inference import Viterbi
import pickle
import pandas as pd
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


def optimize(mat, list_of_mats, weights_path, lamda=10):
    weights, likelihood = Optimization.optimize_weights(mat, list_of_mats, lamda=lamda)
    with open(weights_path, 'wb') as f:
        pickle.dump(weights, f)
    return likelihood, weights


def viterbi_test(test_path, features, weights):
    viterbi = Viterbi(features, weights)
    list_of_sentences, real_list_of_tags = evaluation.prepare_test_data(test_path)
    pred_list_of_tags = viterbi.predict_tags(list_of_sentences)
    for i in range(len(pred_list_of_tags)):
        print("SENTENCE", i)
        print(list_of_sentences[i])
        for word, t1, t2 in zip(list_of_sentences[i].split(' '), real_list_of_tags[i], pred_list_of_tags[i]):
            if t1 != t2:
                print(word, t1, t2)
    accuracy, accuracies, confusion_matrix = evaluation.calculate_accuracy(real_list_of_tags, pred_list_of_tags)
    return accuracy, accuracies, confusion_matrix


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


def train_test_split(data_path, test_size, train_output_path, test_output_path, seed=None):
    lines = open(data_path).readlines()
    test_n = int(len(lines) * test_size)
    random.Random(seed).shuffle(lines)
    test_lines = lines[:test_n]
    train_lines = lines[test_n:]
    open(train_output_path, 'w').writelines(train_lines)
    open(test_output_path, 'w').writelines(test_lines)


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


if __name__ == '__main__':
    model_i = 2
    if model_i == 1:
        train_path = "data/train1.wtag"
        test_path = "data/test1.wtag"
    else:
        data_path = "data/train2.wtag"
        train_path = f"data/train2_{model_i}.wtag"
        test_path = f"data/test2_{model_i}.wtag"
        train_test_split(data_path, 0.2, train_path, test_path, seed=model_i)
    for experiment, thresholds in experiments.items():
        features_path = 'experiments/' + experiment + f'_features{model_i}.pkl'
        weights_path = 'experiments/' + experiment + f'_weights{model_i}.pkl'
        features, mat, list_of_mats = create_features(thresholds, train_path, features_path)
        likelihood, weights = optimize(mat, list_of_mats, weights_path, thresholds['lamda'])
        accuracy, accuracies, confusion_matrix = viterbi_test(test_path, features, weights)
        pd.DataFrame(confusion_matrix).to_csv('experiments/' + experiment + f'_confusion_matrix{model_i}.pkl')
        print("Model:{} results for experiment: {}, likelihood: {}, accuracy: {}".format(model_i, experiment, likelihood, accuracy))
        print(accuracies)
        with open('experiments/results.txt', 'a') as file:
            file.write("Model:{} results for experiment: {}, likelihood: {}, accuracy: {}\n".format(model_i, experiment, likelihood, accuracy))