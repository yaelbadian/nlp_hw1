from pretraining import Features
from optimization import Optimization
from inference import Viterbi
import pickle
import evaluation
from macros import experiments
import random
from time import time


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


def optimize(mat, list_of_mats, weights_path, lamda=0.5):
    weights, likelihood = Optimization.optimize_weights(mat, list_of_mats, lamda=lamda)
    with open(weights_path, 'wb') as f:
        pickle.dump(weights, f)
    return likelihood, weights


def viterbi_test(test_path, features, weights, beam=10):
    viterbi = Viterbi(features, weights, beam=beam)
    list_of_sentences, real_list_of_tags = evaluation.prepare_test_data(test_path)
    pred_list_of_tags = viterbi.predict_tags(list_of_sentences)
    # for i in range(len(pred_list_of_tags)):
    #     print("SENTENCE", i)
    #     print(list_of_sentences[i])
    #     for word, t1, t2 in zip(list_of_sentences[i].split(' '), real_list_of_tags[i], pred_list_of_tags[i]):
    #         if t1 != t2:
    #             print(word, t1, t2)
    accuracy, accuracies, confusion_matrix = evaluation.calculate_accuracy(real_list_of_tags, pred_list_of_tags)
    return accuracy, accuracies, confusion_matrix


def train_test_split(data_path, test_size, train_output_path, test_output_path, seed=None):
    lines = open(data_path).readlines()
    test_n = int(len(lines) * test_size)
    random.Random(seed).shuffle(lines)
    test_lines = lines[:test_n]
    train_lines = lines[test_n:]
    open(train_output_path, 'w').writelines(train_lines)
    open(test_output_path, 'w').writelines(test_lines)


def main_train(model_i):
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
        t0 = time()
        features, mat, list_of_mats = create_features(thresholds, train_path, features_path)
        pretraining_time = time()
        likelihood, weights = optimize(mat, list_of_mats, weights_path, thresholds['lamda'])
        optimization_time = time()
        accuracy, accuracies, confusion_matrix = viterbi_test(test_path, features, weights)
        inference_time = time()
        print(experiment)
        print('\tPretraining time:', pretraining_time - t0)
        print('\tOptimization time:', optimization_time - pretraining_time)
        print('\tTotal training time:', optimization_time - t0)
        print('\tInference time:', inference_time - optimization_time)
        print('\tTotal Features:', len(weights))
        print('\tFeatures Sizes:', features.size_of_features())
        print('\tTest Accuracy:', accuracy)
        print("Model:{} results for experiment: {}, likelihood: {}, accuracy: {}".format(model_i, experiment, likelihood, accuracy))
        print(accuracies)
        with open('experiments/results.txt', 'a') as file:
            file.write("Model:{} results for experiment: {}, likelihood: {}, accuracy: {}\n".format(model_i, experiment, likelihood, accuracy))

def main_test(model_i, experiment):
    if model_i == 1:
        train_path = "data/train1.wtag"
        test_path = "data/test1.wtag"
    else:
        train_path = f"data/train2_{model_i}.wtag"
        test_path = f"data/test2_{model_i}.wtag"
    features_path = 'models/' + experiment + f'_features{model_i}.pkl'
    weights_path = 'models/' + experiment + f'_weights{model_i}.pkl'
    t0 = time()
    features = Features.load(features_path)
    with open(weights_path, 'rb') as file:
        weights = pickle.load(file)
    train_accuracy, _, _ = viterbi_test(train_path, features, weights)
    train_inference_time = time()
    test_accuracy, _, _ = viterbi_test(test_path, features, weights)
    test_inference_time = time()
    print('\tTotal Features:', len(weights))
    print('\tFeatures Sizes:', features.size_of_features())
    print('\tWeights Norm:', (weights ** 2).sum() ** 0.5)
    print('\tTrain Inference time:', train_inference_time - t0)
    print('\tTest Inference time:', test_inference_time - train_inference_time)
    print('\tTrain Accuracy:', train_accuracy)
    print('\tTest Accuracy:', test_accuracy)


if __name__ == '__main__':
    pass
    # model_i = 2
    # experiment = 'exp_2'
    # main_test(model_i, experiment)
    # main_train(model_i)