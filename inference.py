import numpy as np
import scipy
from scipy.sparse import csr_matrix
from collections import defaultdict, OrderedDict

class Viterbi:
    def __init__(self, features, weights):
        self.features = features
        self.weights = weights
        self.qs = {}
        self.denominators = {}

    def calculate_q_numerator(self, history):
        features_indices = self.features.represent_input_with_features(history)
        return np.exp(sum([self.weights[i] for i in features_indices]))

    @staticmethod
    def replace_tag_in_history(history, tag):
        tmp_history = list(history).copy()
        tmp_history[3] = tag
        return tuple(tmp_history)

    def calculate_q_denominator(self, history):
        sum = 0
        for tag in self.features.all_tags:
            tmp_history = self.replace_tag_in_history(history, tag)
            sum += self.calculate_q_numerator(tmp_history)
        tmp_history = self.replace_tag_in_history(history, '')
        self.denominators[tmp_history] = sum
        return sum

    def calculate_q(self, history):
        if history in self.qs:
            return self.qs[history]
        else:
            numerator = self.calculate_q_numerator(history)
            tmp_history = self.replace_tag_in_history(history, '')
            if tmp_history in self.denominators:
                denominator = self.denominators[tmp_history]
            else:
                denominator = self.calculate_q_denominator(history)
            q = numerator / denominator
            self.qs[history] = q
        return q

    def viterbi(self, sentence, b):
        sentence = ['*'] + sentence.rstrip('\n').split(' ') + ['STOP']
        n = len(sentence) - 2
        tags = ['*'] + [''] * n
        pi = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        pi[0]['*']['*'] = 1
        bp = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        for i in range(1, n+1):
            all_scores = []
            for u in pi[i-1].keys():  # all the v's from last iteration
                for v in self.features.all_tags:  # all the v's from current iteration
                    best_pi_ivu_score = 0
                    best_bp_ivu = ''
                    for t in pi[i-1][u].keys():
                        history = (sentence[i], t, u, v, sentence[i+1], sentence[i-1])
                        score = pi[i-1][u][t] * self.calculate_q(history)
                        if score >= best_pi_ivu_score:
                            best_pi_ivu_score = score
                            best_bp_ivu = t
                    all_scores.append([best_pi_ivu_score, best_bp_ivu, v, u])
            for pi_score, pi_bp, pi_v, pi_u in sorted(all_scores, key=lambda x: x[0], reverse=True)[:b]:
                pi[i][pi_v][pi_u] = pi_score
                bp[i][pi_v][pi_u] = pi_bp
                # print("word:", sentence[i], "i:", i, "u:", pi_u, "v:", pi_v, "t:", bp[i][pi_v][pi_u], "pi:", pi[i][pi_v][pi_u])
        score = 0
        for v in pi[n].keys():
            for u in pi[n][v].keys():
                if score < pi[n][v][u]:
                    score = pi[n][v][u]
                    tags[n] = v
                    tags[n-1] = u
        for i in range(n-2, 0, -1):
            tags[i] = bp[i+2][tags[i+2]][tags[i+1]]
        return tags[1:]

    def predict_tags(self, list_of_sentences):
        list_of_tags = []
        for sentence in list_of_sentences:
            list_of_tags.append(self.viterbi(sentence, 30))
        return list_of_tags




