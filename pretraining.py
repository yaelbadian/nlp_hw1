import numpy as np
import scipy
from collections import OrderedDict


class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.word_ctag_count = OrderedDict()  # feature 100
        self.suffix_count = OrderedDict()  # feature 101
        self.prefix_count = OrderedDict()  # feature 102
        self.pptag_ptag_ctag_count = OrderedDict()  # feature 103
        self.ptag_ctag_count = OrderedDict()  # feature 104
        self.ctag_count = OrderedDict()  # feature 105
        self.pword_ctag_count = OrderedDict()  # feature 106
        self.nword_ctag_count = OrderedDict()  # feature 107
        self.len_word_count = OrderedDict()  # len of the word



    def add_key_to_dict(self, key, dict):
        if key in dict:
            dict[key] += 1
        else:
            dict[key] = 1

    def fill_word_ctag_count(self, word, ctag):
        key = (word, ctag)
        self.add_key_to_dict(key, self.word_ctag_count)

    def fill_suffix_count(self, word, ctag):
        n = len(word)
        for i in range(1, 4):
            if n >= i + 2:
                key = (word[-i:], ctag)
                self.add_key_to_dict(key, self.suffix_count)

    def fill_prefix_count(self, word, ctag):
        n = len(word)
        for i in range(1, 4):
            if n >= i + 2:
                key = (word[:i], ctag)
                self.add_key_to_dict(key, self.prefix_count)

    def fill_pptag_ptag_ctag_count(self, pptag, ptag, ctag):
        key = (pptag, ptag, ctag)
        self.add_key_to_dict(key, self.pptag_ptag_ctag_count)

    def fill_ptag_ctag_count(self, ptag, ctag):
        key = (ptag, ctag)
        self.add_key_to_dict(key, self.ptag_ctag_count)

    def fill_ctag_count(self, ctag):
        key = (ctag)
        self.add_key_to_dict(key, self.ctag_count)

    def fill_pword_ctag_count(self, pword, ctag):
        key = (pword, ctag)
        self.add_key_to_dict(key, self.pword_ctag_count)

    def fill_nword_ctag_count(self, nword, ctag):
        key = (nword, ctag)
        self.add_key_to_dict(key, self.nword_ctag_count)

    def fill_all_dicts(self, history):
        word, pptag, ptag, ctag, nword, pword = history
        self.fill_word_ctag_count(word, ctag)
        self.fill_suffix_count(word, ctag)
        self.fill_pptag_ptag_ctag_count(pptag, ptag, ctag)
        self.fill_ptag_ctag_count(ptag, ctag)
        self.fill_ctag_count(ctag)
        self.fill_pword_ctag_count(pword, ctag)
        self.fill_nword_ctag_count(nword, ctag)

    @staticmethod
    def create_histories(line):
        splitted_line = line.rstrip('\n').split(' ')
        list_of_histories = []
        list_of_tuples = [['*', '*'], ['*', '*']] + [x.split('_') for x in splitted_line] + [['STOP', 'STOP']]
        for i in range(2, len(list_of_tuples)-1):
            list_of_histories.append([list_of_tuples[i][0], list_of_tuples[i-2][1], list_of_tuples[i-1][1],
                                      list_of_tuples[i][1], list_of_tuples[i+1][0], list_of_tuples[i-1][0]])
        return list_of_histories

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                line_histories = self.create_histories(line)
                for history in line_histories:
                    self.fill_all_dicts(history)

        print(self.suffix_count)

file_path = "data/train1.wtag"
features = feature_statistics_class()
features.get_word_tag_pair_count(file_path)
