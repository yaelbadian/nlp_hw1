from collections import OrderedDict, Counter, defaultdict
from scipy.sparse import csr_matrix
import pickle
from macros import pucts, numbers, prefixes, suffixes, simple_past, past_participle

class Features:

    def __init__(self, file_path, thresholds):
        self.thresholds = thresholds
        self.n_total_features = 0  # Total number of features accumulated
        self.n_total_histories = 0
        self.all_tags = set([])
        self.pucts = pucts
        self.numbers = numbers
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.simple_past = simple_past
        self.past_participle = past_participle
        self.list_of_lines_histories = self.create_list_of_lines_histories(file_path)

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
        self.is_number_count = OrderedDict() # True if word is number, else False
        self.irregular_verb_count = OrderedDict() # irregular verbs
        self.idx_to_feature = {}
        self.create_all_dicts()

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
            mapped = Features.map_char(c)
            if mapped == 0:
                has_upper = True
            elif mapped == 1:
                has_lower = True
            elif mapped == 2:
                has_number = True
        return has_upper, has_lower, has_number

    @staticmethod
    def add_key_to_dict(key, dict):
        if key in dict:
            dict[key] += 1
        else:
            dict[key] = 1

    def fill_word_ctag_count(self, word, ctag, fill=True):
        key = (word.lower(), ctag)
        if fill:
            self.add_key_to_dict(key, self.word_ctag_count)
        else:
            if key in self.word_ctag:
                return [self.word_ctag[key]]
            else:
                return []

    def fill_suffix_count(self, word, ctag, fill=True):
        features = []
        word = word.lower()
        n = len(word)
        for i in range(7, 0, -1):
            if i >= n:
                continue
            if word[-i:] in self.suffixes:
                key = (word[-i:], ctag)
                if fill:
                    self.add_key_to_dict(key, self.suffix_count)
                else:
                    if key in self.suffix:
                        features.append(self.suffix[key])
                break
        if not fill:
            return features
        else:
            return []

    def fill_prefix_count(self, word, ctag, fill=True):
        features = []
        word = word.lower()
        n = len(word)
        for i in range(7, 0, -1):
            if word[:i] in self.prefixes:
                if i >= n:
                    continue
                key = (word[:i], ctag)
                if fill:
                    self.add_key_to_dict(key, self.prefix_count)
                else:
                    if key in self.prefix:
                        features.append(self.prefix[key])
                break
        if not fill:
            return features
        else:
            return []

    def fill_pptag_ptag_ctag_count(self, pptag, ptag, ctag, fill=True):
        key = (pptag, ptag, ctag)
        if fill:
            self.add_key_to_dict(key, self.pptag_ptag_ctag_count)
        else:
            if key in self.pptag_ptag_ctag:
                return [self.pptag_ptag_ctag[key]]
            else:
                return []

    def fill_ptag_ctag_count(self, ptag, ctag, fill=True):
        key = (ptag, ctag)
        if fill:
            self.add_key_to_dict(key, self.ptag_ctag_count)
        else:
            if key in self.ptag_ctag:
                return [self.ptag_ctag[key]]
            else:
                return []

    def fill_ctag_count(self, ctag, fill=True):
        key = (ctag)
        if fill:
            self.add_key_to_dict(key, self.ctag_count)
        else:
            if key in self.ctag:
                return [self.ctag[key]]
            else:
                return []

    def fill_pword_ctag_count(self, pword, ctag, fill=True):
        key = (pword.lower(), ctag)
        if fill:
            self.add_key_to_dict(key, self.pword_ctag_count)
        else:
            if key in self.pword_ctag:
                return [self.pword_ctag[key]]
            else:
                return []

    def fill_nword_ctag_count(self, nword, ctag, fill=True):
        key = (nword.lower(), ctag)
        if fill:
            self.add_key_to_dict(key, self.nword_ctag_count)
        else:
            if key in self.nword_ctag:
                return [self.nword_ctag[key]]
            else:
                return []

    def fill_len_word_count(self, word, ctag, fill=True):
        key = (len(word), ctag)
        if fill:
            self.add_key_to_dict(key, self.len_word_count)
        else:
            if key in self.len_word:
                return [self.len_word[key]]
            else:
                return []

    def fill_upper_lower_number_count(self, word, ctag, ptag, fill=True):
        c = Features.map_char(word[0])
        if ptag == '*' and c == 0:
            c = 1
        has_upper, has_lower, has_number = Features.map_word(word[1:])
        if c == 2 and len(word) == 1:
            has_number = True
        key = (c, has_upper, has_lower, has_number, ctag)
        if fill:
            self.add_key_to_dict(key, self.upper_lower_number_count)
        else:
            if key in self.upper_lower_number:
                return [self.upper_lower_number[key]]
            else:
                return []
        return []

    def fill_punctuation_starts_count(self, word, ctag, fill=True):
        key = None
        word = word.replace('.', '')
        for c in word:
            if Features.map_char(c) == 3:
                mapped = word.split(c)[-1]
                if len(mapped) > 0:  # ends with punc
                    mapped = Features.map_char(word.split(c)[-1][0])
                else:
                    mapped = 3
                key = (c, Features.map_char(word[0]), mapped, ctag)
        if fill and key is not None:
            self.add_key_to_dict(key, self.punctuation_starts_count)
            return
        elif not fill:
            if key in self.punctuation_starts:
                return [self.punctuation_starts[key]]
            else:
                return []

    def fill_punctuation_count(self, word, ctag, fill=True):
        key = None
        word = word.replace('.', '')
        for c in word:
            if Features.map_char(c) == 3:
                has_upper, has_lower, has_number = Features.map_word(word)
                key = (c, has_upper, has_lower, has_number, ctag)
        if fill and key is not None:
            self.add_key_to_dict(key, self.punctuation_count)
            return
        elif not fill:
            if key in self.punctuation:
                return [self.punctuation[key]]
            else:
                return []

    def fill_num_of_uppers_count(self, word, ctag, fill=True):
        n = 0
        for c in word:
            if Features.map_char(c) == 0:
                n += 1
        key = (n, ctag)
        if fill:
            self.add_key_to_dict(key, self.num_of_uppers_count)
        else:
            if key in self.num_of_uppers:
                return [self.num_of_uppers[key]]
            else:
                return []

    def fill_is_number_count(self, word, ctag, fill=True):
        is_number = True
        if word[0] == '-':
            word = word[1:]
        word = word.replace('.', '').replace(',', '')
        for c in word:
            if self.map_char(c) != 2:
                is_number = False
        if word.lower() in self.numbers:
            is_number = True
        key = (is_number, ctag)
        if fill:
            self.add_key_to_dict(key, self.is_number_count)
        else:
            if key in self.is_number:
                return [self.is_number[key]]
            else:
                return []

    def fill_irregular_verb_count(self, word, ctag, fill=True):
        key = None
        if word in self.simple_past and word in self.past_participle:
            key = (2, ctag)
        elif word in self.simple_past:
            key = (0, ctag)
        elif word in self.past_participle:
            key = (1, ctag)
        if fill and key is not None:
            self.add_key_to_dict(key, self.irregular_verb_count)
        else:
            if key is not None and key in self.irregular_verb:
                return [self.irregular_verb[key]]
            else:
                return []

    def fill_all_dicts(self, history):
        word, pptag, ptag, ctag, nword, pword = history
        self.fill_word_ctag_count(word, ctag)
        self.fill_pword_ctag_count(pword, ctag)
        self.fill_nword_ctag_count(nword, ctag)
        self.fill_len_word_count(word, ctag)
        self.fill_is_number_count(word, ctag)
        if len(word) > 1:
            self.fill_suffix_count(word, ctag)
            self.fill_prefix_count(word, ctag)
            self.fill_ctag_count(ctag)
            self.fill_pptag_ptag_ctag_count(pptag, ptag, ctag)
            self.fill_ptag_ctag_count(ptag, ctag)
            self.fill_upper_lower_number_count(word, ctag, ptag)
            self.fill_punctuation_starts_count(word, ctag)
            self.fill_punctuation_count(word, ctag)
            self.fill_num_of_uppers_count(word, ctag)
            self.fill_irregular_verb_count(word, ctag)

    def create_idx_dict(self, dict_count, threshold):
        dict_idx = OrderedDict()
        for key in dict_count:
            if dict_count[key] > threshold:
                dict_idx[key] = self.n_total_features
                self.idx_to_feature[self.n_total_features] = key
                self.n_total_features += 1
        return dict_idx

    def create_all_idx_dicts(self):
        self.word_ctag = self.create_idx_dict(self.word_ctag_count, self.thresholds['word_ctag'])
        self.suffix = self.create_idx_dict(self.suffix_count, self.thresholds['suffix'])
        self.prefix = self.create_idx_dict(self.prefix_count, self.thresholds['prefix'])
        self.pptag_ptag_ctag = self.create_idx_dict(self.pptag_ptag_ctag_count, self.thresholds['pptag_ptag_ctag'])
        self.ptag_ctag = self.create_idx_dict(self.ptag_ctag_count, self.thresholds['ptag_ctag'])
        self.ctag = self.create_idx_dict(self.ctag_count, self.thresholds['ctag'])
        self.pword_ctag = self.create_idx_dict(self.pword_ctag_count, self.thresholds['pword_ctag'])
        self.nword_ctag = self.create_idx_dict(self.nword_ctag_count, self.thresholds['nword_ctag'])
        self.len_word = self.create_idx_dict(self.len_word_count, self.thresholds['len_word'])
        self.upper_lower_number = self.create_idx_dict(self.upper_lower_number_count, self.thresholds['upper_lower_number'])
        self.punctuation_starts = self.create_idx_dict(self.punctuation_starts_count, self.thresholds['punctuation_starts'])
        self.punctuation = self.create_idx_dict(self.punctuation_count, self.thresholds['punctuation'])
        self.num_of_uppers = self.create_idx_dict(self.num_of_uppers_count, self.thresholds['num_of_uppers'])
        self.is_number = self.create_idx_dict(self.is_number_count, self.thresholds['is_number'])
        self.irregular_verb = self.create_idx_dict(self.irregular_verb_count, self.thresholds['irregular_verb'])

    @staticmethod
    def create_histories(line):
        splitted_line = line.rstrip('\n').split(' ')
        list_of_histories = []
        list_of_tuples = [['*', '*'], ['*', '*']] + [x.split('_') for x in splitted_line] + [['STOP', 'STOP']]
        for i in range(2, len(list_of_tuples)-1):
            list_of_histories.append([list_of_tuples[i][0], list_of_tuples[i-2][1], list_of_tuples[i-1][1],
                                      list_of_tuples[i][1], list_of_tuples[i+1][0], list_of_tuples[i-1][0]])
        return list_of_histories

    def represent_input_with_features(self, history):
        word, pptag, ptag, ctag, nword, pword = history
        features = []
        features += self.fill_word_ctag_count(word, ctag, fill=False)
        features += self.fill_pword_ctag_count(pword, ctag, fill=False)
        features += self.fill_nword_ctag_count(nword, ctag, fill=False)
        features += self.fill_len_word_count(word, ctag, fill=False)
        features += self.fill_is_number_count(word, ctag, fill=False)
        if len(word) > 1:  ########## think about punctuation ############
            features += self.fill_suffix_count(word, ctag, fill=False)
            features += self.fill_prefix_count(word, ctag, fill=False)
            features += self.fill_pptag_ptag_ctag_count(pptag, ptag, ctag, fill=False)
            features += self.fill_ptag_ctag_count(ptag, ctag, fill=False)
            features += self.fill_ctag_count(ctag, fill=False)
            features += self.fill_upper_lower_number_count(word, ctag, ptag, fill=False)
            features += self.fill_punctuation_starts_count(word, ctag, fill=False)
            features += self.fill_punctuation_count(word, ctag, fill=False)
            features += self.fill_num_of_uppers_count(word, ctag, fill=False)
            features += self.fill_irregular_verb_count(word, ctag, fill=False)
        return features

    def create_list_of_lines_histories(self, file_path):
        list_of_lines_histories = []
        with open(file_path) as f:
            for line in f:
                list_of_lines_histories.append(self.create_histories(line))
        self.all_tags = set([h[3] for line_histories in list_of_lines_histories for h in line_histories])
        return list_of_lines_histories

    def create_all_dicts(self):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        for line_histories in self.list_of_lines_histories:
            for history in line_histories:
                self.n_total_histories += 1
                self.fill_all_dicts(history)
        self.create_all_idx_dicts()

    def create_features(self, tag=''):
        idx = 0
        row, col, data = [], [], []
        for line_histories in self.list_of_lines_histories:
            for history in line_histories:
                history_tmp = history.copy()
                if tag != '':
                    history_tmp[3] = tag
                features = self.represent_input_with_features(history_tmp)
                n_feature = len(features)
                row += [idx] * n_feature
                col += features
                data += [True] * n_feature
                idx += 1
        return csr_matrix((data, (row, col)), shape=(self.n_total_histories, self.n_total_features), dtype=bool)

    def create_all_mats(self):
        list_of_mats = []
        tags = list(self.all_tags)
        for tag in tags:
            list_of_mats.append(self.create_features(tag))
        return list_of_mats

    @staticmethod
    def print_statistics(name, dic):
        print('##', name, '##')
        print("# keys+ctag: ", len(dic))
        print("sum of values: ", sum([dic[key] for key in dic]))
        print("# keys wiothout ctag: ", len(set([key[:-1] for key in dic])))
        print("check threshold: ")
        for i in range(1, 5):
            print(i, len([key for key in dic if dic[key] >= i]))
        print("most common keys: ", Counter(dic).most_common(50))

    def save(self, fname):
        with open(fname, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def calculate_statistics_df(dic):
        import numpy as np
        import pandas as pd
        rows = defaultdict(lambda: defaultdict(int))
        ctag_rows = defaultdict(lambda: defaultdict(int))
        for key_ctag, cnt in dic.items():
            key, ctag = str(key_ctag[:-1]), str(key_ctag[-1])
            rows[key][ctag] = cnt
            ctag_rows[ctag]['nunique'] += 1
            ctag_rows[ctag]['total'] += cnt
        for key in rows:
            total, entropy = 0, 0
            for ctag in rows[key]:
                total += rows[key][ctag]
            for ctag in rows[key]:
                rows[key][ctag] = rows[key][ctag] / max(1, total)
                entropy += -rows[key][ctag] * np.log(rows[key][ctag])
            rows[key]['nunique'] = len(rows[key])
            rows[key]['total'] = total
            rows[key]['entropy'] = entropy
        df = pd.DataFrame(rows).T.fillna(0)
        ctags_df = pd.DataFrame(ctag_rows)
        ctags_entropy = pd.DataFrame(df.apply(lambda x: (- x * np.log(x.where(x > 0, 1))).sum(), axis=0).to_dict(),
                                     index=['entropy'])
        return pd.concat([df, ctags_df, ctags_entropy], axis=0)

    def save_statistics_for_all_dict(self, file_prefix):
        for dic_name, dic in [('word_ctag', self.word_ctag_count), ('suffix', self.suffix_count), ('prefix', self.prefix_count),
                              ('pptag_ptag_ctag', self.pptag_ptag_ctag_count), ('ptag_ctag', self.ptag_ctag_count),
                              ('ctag', self.ctag_count), ('pword_ctag', self.pword_ctag_count), ('nword_ctag', self.nword_ctag_count),
                              ('len_word', self.len_word_count), ('upper_lower_number', self.upper_lower_number_count),
                              ('punctuation_starts', self.punctuation_starts_count ), ('punctuation', self.punctuation_count),
                              ('num_of_uppers', self.num_of_uppers_count), ('is_number', self.is_number_count)]:
            self.calculate_statistics_df(dic).to_csv(file_prefix + '_' + dic_name + '.csv')