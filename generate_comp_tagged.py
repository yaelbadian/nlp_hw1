from pretraining import Features
from inference import Viterbi
import pickle
import evaluation

features_model1 = 'models/exp_5_features.pkl'
weights_model1 = 'models/exp_5_weights.pkl'
comp_input_model1 = 'data/comp1.words'
comp_output_model1 = 'data/comp_m1_204434161.wtag'
features_model2 = 'models/exp_5_features2.pkl'
weights_model2 = 'models/exp_5_weights2.pkl'
comp_input_model2 = 'data/comp2.words'
comp_output_model2 = 'data/comp_m2_204434161.wtag'



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


if __name__ == '__main__':
    # model 1
    features1 = Features.load(features_model1)
    with open(weights_model1, 'rb') as file:
        weights1 = pickle.load(file)
    viterbi_comp(comp_input_model1, features1, weights1, comp_output_model1)
    del features1
    del weights1

    # model 2
    features2 = Features.load(features_model2)
    with open(weights_model2, 'rb') as file:
        weights2 = pickle.load(file)
    viterbi_comp(comp_input_model2, features2, weights2, comp_output_model2)
