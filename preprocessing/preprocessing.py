# from abc_preprocessing import ABCPreProcessing
from . import abc_preprocessing

import sys
sys.path.append("../")
sys.path.append("../../")
import keras
import random as rand
from itertools import chain
import numpy as np

from PIL import Image
from . import config
from cnn_autoencoder import get_feature
from cnn_autoencoder.model.simple_autoencoder import SimpleAutoencoder

num_classes = 10


class PreProcessing(abc_preprocessing.ABCPreProcessing):
    @classmethod
    def __get_word_lists(cls, file_path):
        print("make wordlists")
        with open(file_path) as f:
            lines = f.read().split("\n")
        word_lists = []
        for line in lines:
            word_lists.append(line.split(" "))
        print("wordlist num:", len(word_lists))
        return word_lists[:-1]

    @classmethod
    def __to_uniq(cls, word_lists):
        word_flat_list = list(chain.from_iterable(word_lists))
        word_uniq_lists = list(set(word_flat_list))
        if ' ' in word_uniq_lists:
            word_uniq_lists.remove(' ')
        if '' in word_uniq_lists:
            word_uniq_lists.remove('')
        return word_uniq_lists

    @classmethod
    def make_train_data(cls, data_size, window_size=5):
        autoencoder = SimpleAutoencoder.load_model("cnn_model.hdf5")
        encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

        word_list = PreProcessing.__get_word_lists(
            "../aozora_data/files/files_all_rnp.txt")

        from itertools import chain
        word_list = list(chain.from_iterable(word_list))
        word_list = "".join(word_list)

        window_sentence = []
        train_data = []
        teach_data = []

        rand_num = rand.randint(0, len(word_list) - data_size + 1)
        for char in word_list[rand_num: rand_num + data_size]:
            feature = get_feature.char2feature(char, encoder)
            feature = feature.reshape(4 * 4 * 8)
            window_sentence.append(feature)
            if len(window_sentence) == window_size + 1:
                train_data.append(window_sentence[:-1])
                teach_data.append(window_sentence[1:])
                window_sentence = window_sentence[:-1]

        train_data = np.array(train_data)
        teach_data = np.array(teach_data)

        print("train shape:", train_data.shape)
        print("teach shape:", teach_data.shape)
        return train_data, teach_data

    @classmethod
    def make_test_data(cls, char):
        autoencoder = SimpleAutoencoder.load_model("cnn_model.hdf5")
        encoder = SimpleAutoencoder.make_encoder_model(autoencoder)
        feature = get_feature.char2feature(char, encoder)
        feature = feature.reshape(1, 1, 4 * 4 * 8)
        print(feature.shape)
        return feature


def main():
    pass


if __name__ == '__main__':
    main()
