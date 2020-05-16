from collections import defaultdict


def prepare_test_data(test_path):
    list_of_sentences = []
    list_of_tags = []
    with open(test_path, 'r') as file:
        for line in file:
            sentence = ''
            tags = []
            for word in line.split(' '):
                sentence += word.split('_')[0] + ' '
                tags.append(word.split('_')[1].rstrip())
            list_of_sentences.append(sentence.rstrip())
            list_of_tags.append(tags)
    return list_of_sentences, list_of_tags


def prepare_comp_data(comp_path):
    list_of_sentences = []
    with open(comp_path, 'r') as file:
        for line in file:
            list_of_sentences.append(line)
    return list_of_sentences


def create_confusion_matrix(real_list_of_tags, pred_list_of_tags):
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for real_tags, pred_tags in zip(real_list_of_tags, pred_list_of_tags):
        for real_tag, pred_tag in zip(real_tags, pred_tags):
            confusion_matrix[real_tag][pred_tag] += 1
    return confusion_matrix


def calculate_accuracy(real_list_of_tags, pred_list_of_tags):
    confusion_matrix = create_confusion_matrix(real_list_of_tags, pred_list_of_tags)
    accuracy = 0
    total = 0
    accuracies = {}
    for real_tag in confusion_matrix:
        sum_real_tag = 0
        accuracy_real_tag = 0
        for pred_tag in confusion_matrix[real_tag]:
            total += confusion_matrix[real_tag][pred_tag]
            sum_real_tag += confusion_matrix[real_tag][pred_tag]
            if real_tag == pred_tag:
                accuracy += confusion_matrix[real_tag][real_tag]
                accuracy_real_tag += confusion_matrix[real_tag][real_tag]
        accuracies[real_tag] = accuracy_real_tag / sum_real_tag
    return accuracy / total, accuracies, confusion_matrix
