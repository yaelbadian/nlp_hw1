import numpy as np
import scipy
from collections import OrderedDict


class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated
        self.pucts = ['!', '@', '#', '.', ':', ',', '$', '&', '%', '$', '~', "'", '+', '=', '*', '^', '>', '<', ';', '``']

        # Init all features dictionaries
        self.word_ctag_count = OrderedDict()  # feature 100
        self.suffix_count = OrderedDict()  # feature 101
        self.prefix_count = OrderedDict()  # feature 102
        self.pptag_ptag_ctag_count = OrderedDict()  # feature 103
        self.ptag_ctag_count = OrderedDict()  # feature 104
        self.ctag_count = OrderedDict()  # feature 105
        self.pword_ctag_count = OrderedDict()  # feature 106
        self.nword_ctag_count = OrderedDict()  # feature 107

        self.len_word_count = OrderedDict()  # length of the word
        self.upper_lower_number_count = OrderedDict()  # (1) starts with 0-Upper 1-Lower 2-number (2) have uppers (3) have lowers (4) have numbers
        self.punctuation_starts_count = OrderedDict()  # (1) punc (2) starts with 0-Upper 1-Lower 2-number (3) after punc starts with 0-Upper 1-Lower 2-number
        self.punctuation_count = OrderedDict()  # (1) punc (2) have uppers (3) have lowers (4) have numbers
        self.num_of_uppers_count = OrderedDict()  # numbers of uppers


    @staticmethod
    def map_char(c):
        if c.isupper():
            return 0
        elif c.islower():
            return 1
        elif c.isnumeric():
            return 2
        else:  # punc
            return 3

    @staticmethod
    def map_word(word):
        has_upper, has_lower, has_number = False, False, False
        for c in word:
            mapped = feature_statistics_class.map_char(c)
            if mapped == 0:
                has_upper = True
            elif mapped == 1:
                has_lower = True
            elif mapped == 2:
                has_number = True
        return has_upper, has_lower, has_number

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

    def fill_len_word_count(self, word, ctag):
        key = (len(word), ctag)
        self.add_key_to_dict(key, self.len_word_count)

    def fill_upper_lower_number_count(self, word, ctag):
        c = word[0]
        has_upper, has_lower, has_number = feature_statistics_class.map_word(word[1:])
        key = (feature_statistics_class.map_char(c), has_upper, has_lower, has_number, ctag)
        self.add_key_to_dict(key, self.upper_lower_number_count)

    def fill_punctuation_starts_count(self, word, ctag):
        word = word.replace('.', '')
        for c in word:
            if feature_statistics_class.map_char(c) == 3:
                mapped = word.split(c)[-1]
                if len(mapped) > 0:  # ends with punc
                    mapped = feature_statistics_class.map_char(word.split(c)[-1][0])
                else:
                    mapped = 3
                key = (c, feature_statistics_class.map_char(word[0]), mapped, ctag)
                self.add_key_to_dict(key, self.punctuation_starts_count)
                return

    def fill_punctuation_count(self, word, ctag):
        word = word.replace('.', '')
        for c in word:
            if feature_statistics_class.map_char(c) == 3:
                has_upper, has_lower, has_number = feature_statistics_class.map_word(word)
                key = (c, has_upper, has_lower, has_number, ctag)
                self.add_key_to_dict(key, self.punctuation_count)
                return

    def fill_num_of_uppers_count(self, word, ctag):
        n = 0
        for c in word:
            if feature_statistics_class.map_char(c) == 0:
                n += 1
        key = (n, ctag)
        self.add_key_to_dict(key, self.num_of_uppers_count)

    def fill_all_dicts(self, history):
        word, pptag, ptag, ctag, nword, pword = history
        self.fill_word_ctag_count(word, ctag)
        if len(word) > 1:  ########## think about punctuation ############
            self.fill_suffix_count(word, ctag)
            self.fill_prefix_count(word, ctag)
            self.fill_pptag_ptag_ctag_count(pptag, ptag, ctag)
            self.fill_ptag_ctag_count(ptag, ctag)
            self.fill_ctag_count(ctag)
            self.fill_pword_ctag_count(pword, ctag)
            self.fill_nword_ctag_count(nword, ctag)
            self.fill_len_word_count(word, ctag)
            self.fill_upper_lower_number_count(word, ctag)
            self.fill_punctuation_starts_count(word, ctag)
            self.fill_punctuation_count(word, ctag)
            self.fill_num_of_uppers_count(word, ctag)


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




file_path = "data/train1.wtag"
features = feature_statistics_class()
features.get_word_tag_pair_count(file_path)
