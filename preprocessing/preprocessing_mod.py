import sys
sys.path.append("../")
sys.path.append("../../")
import keras
import random as rand
from itertools import chain
import numpy as np

from PIL import Image
from preprocessing.abc_preprocessing import ABCPreProcessing
from preprocessing.config import Config
from cnn_autoencoder import get_feature
from cnn_autoencoder.model.simple_autoencoder import SimpleAutoencoder


class PreProcessing(ABCPreProcessing):
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
            Config.up_two_dir + "aozora_data/files/files_all_rnp.txt")

        from itertools import chain
        word_list = list(chain.from_iterable(word_list))
        word_list = "".join(word_list)

        window_sentence = []
        train_data = []
        teach_data = []
        rand_num = rand.randint(0, len(word_list) - data_size + 1)

        for char in word_list[rand_num: rand_num + data_size]:
            sys.stdout.write("\r now:(%d/%d)" % (len(train_data), data_size))
            sys.stdout.flush()
            feature = get_feature.char2feature(char, encoder)
            feature = feature.reshape(4 * 4 * 8)
            window_sentence.append(feature)
            if len(window_sentence) == window_size + 1:
                train_data.append(window_sentence[:-1])
                teach_data.append(window_sentence[1:])
                window_sentence = window_sentence[1:]
        print("")
        train_data = np.array(train_data)
        teach_data = np.array(teach_data)

        print("train shape:", train_data.shape)
        print("teach shape:", teach_data.shape)
        return train_data, teach_data

    @classmethod
    def save_train_data(cls, data_size, window_size=5):
        autoencoder = SimpleAutoencoder.load_model("cnn_model.hdf5")
        encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

        word_list = PreProcessing.__get_word_lists(
            Config.up_two_dir + "aozora_data/files/files_all_rnp.txt")

        from itertools import chain
        word_list = list(chain.from_iterable(word_list))
        word_list = "".join(word_list)

        window_sentence = []
        train_data = []
        teach_data = []
        print("window size: ", window_size)
        print("number of chars:", len(word_list))
        if len(word_list) < data_size:
            print("error! data_size is over number of char!")
            exit(0)

        for char in word_list[0: data_size]:
            sys.stdout.write("\r now:(%d/%d)" % (len(train_data), data_size))
            sys.stdout.flush()
            feature = get_feature.char2feature(char, encoder)
            feature = feature.reshape(4 * 4 * 8)
            window_sentence.append(feature)
            if len(window_sentence) == window_size + 1:
                train_data.append(window_sentence[:-1])
                teach_data.append(window_sentence[1:])
                window_sentence = window_sentence[1:]
        print("")
        train_data = np.array(train_data)
        teach_data = np.array(teach_data)
        print("train shape:", train_data.shape)
        print("teach shape:", teach_data.shape)
        print("save")
        np.save(Config.run_dir_path + "/train-" +
                str(data_size) + "-" + str(window_size), train_data)
        np.save(Config.run_dir_path + "/teach-" +
                str(data_size) + "-" + str(window_size), teach_data)

    @classmethod
    def save_split_train_data(cls, split_num=5, window_size=5):
        """
        train dataがでかすぎて1つのファイルに乗らないので分割して保存
        """

        autoencoder = SimpleAutoencoder.load_model("cnn_model.hdf5")
        encoder = SimpleAutoencoder.make_encoder_model(autoencoder)

        word_list = PreProcessing.__get_word_lists(
            Config.up_two_dir + "aozora_data/files/files_all_rnp_head20000.txt")

        from itertools import chain
        word_list = list(chain.from_iterable(word_list))
        word_list = "".join(word_list)

        window_sentence = []
        one_set_size = int(len(word_list) / split_num)
        print("window size: ", window_size)
        print("number of chars:", len(word_list))
        print("one set size: ", one_set_size)

        loop_arr = [[i * one_set_size, (i + 1) * one_set_size]
                    for i in range(split_num)]

        for v in loop_arr:
            start = v[0]
            end = v[1]
            train_data = []
            teach_data = []
            print("loop: " + str(start) + "-" + str(end))
            for char in word_list[start:end]:
                sys.stdout.write("\r now:(%d/%d)" %
                                 (len(train_data), len(word_list)))
                sys.stdout.flush()
                feature = get_feature.char2feature(char, encoder)
                feature = feature.reshape(4 * 4 * 8)
                window_sentence.append(feature)
                if len(window_sentence) == window_size + 1:
                    train_data.append(window_sentence[:-1])
                    teach_data.append(window_sentence[1:])
                    window_sentence = window_sentence[1:]
            print("")
            train_data = np.array(train_data)
            teach_data = np.array(teach_data)
            print("train shape:", train_data.shape)
            print("teach shape:", teach_data.shape)
            print("save")
            np.savez_compressed(Config.run_dir_path + "/train-" +
                                str(start) + "-" + str(end) + "-" + str(window_size), train_data)
            np.savez_compressed(Config.run_dir_path + "/teach-" +
                                str(start) + "-" + str(end) + "-" + str(window_size), teach_data)

    @classmethod
    def load_split_train_data(cls, prefix):
        train_data = np.load(Config.run_dir_path + "/train-" + prefix)
        train_key = train_data.keys()[-1]
        teach_data = np.load(Config.run_dir_path + "/teach-" + prefix)
        teach_key = teach_data.keys()[-1]
        print("train shape:", train_data[train_key].shape)
        print("teach shape:", teach_data[teach_key].shape)
        return train_data[train_key], teach_data[teach_key]

    @classmethod
    def load_train_data(cls):
        train_data = np.load(Config.run_dir_path + "/train-60000-25.npy")
        teach_data = np.load(Config.run_dir_path + "/teach-60000-25.npy")
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
    arg = sys.argv[-1]
    if arg == "save":
        # PreProcessing.save_train_data(60000, window_size=25)
        PreProcessing.save_split_train_data(split_num=120, window_size=25)
    elif arg == "load":
        PreProcessing.load_train_data()
    else:
        print("set args [save] or [load]")
        exit(0)


if __name__ == '__main__':
    main()
