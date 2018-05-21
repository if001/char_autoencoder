

from model.char_autoencoder import CharAutoencoder

from preprocessing.preprocessing_mod import PreProcessing
from model_exec.predict import Predict
import numpy as np

import sys
sys.path.append("../")
from string2image.img2str import Image2String
from cnn_autoencoder.model.simple_autoencoder import SimpleAutoencoder


def one_sentence(start_char, char_model, decoder_model):
    feature_list = PreProcessing().make_test_data(start_char)
    loop = True
    while(loop):
        predict_feature_list = Predict.run(char_model, feature_list)
        last_feature = np.array(
            predict_feature_list[0][-1]).reshape(1, 1, 128)
        feature_list = np.append(feature_list, last_feature, axis=1)
        if feature_list.shape[1] == 5:
            loop = False

    sentence = ""

    for feature in feature_list[0]:
        feature = feature.reshape(1, 4, 4, 8)
        img = Predict.run(decoder_model, feature)
        char = Image2String.image2string(img)
        print(char)
        sentence += char
    return sentence


def main():
    char_model = CharAutoencoder().load_model()
    cnn_model = SimpleAutoencoder().load_model("cnn_model.hdf5")
    decoder_model = SimpleAutoencoder().make_decoder_model(cnn_model)

    char_list = ["B", "S", "私", "こ", "明", "探"]
    for start_char in char_list:
        s = one_sentence(start_char, char_model, decoder_model)
        print(s)


def test():
    char_model = CharAutoencoder().load_model()
    cnn_model = SimpleAutoencoder().load_model("cnn_model.hdf5")
    decoder_model = SimpleAutoencoder().make_decoder_model(cnn_model)

    test_sentence = "今晩の話手と定められた新入"
    feature_list = []
    for char in test_sentence:
        tmp = PreProcessing().make_test_data(char)
        feature_list.append(list(tmp))

    feature_list = np.array(feature_list).reshape(1, len(test_sentence), 128)
    print(feature_list.shape)

    loop = True
    while(loop):
        predict_feature_list = Predict.run(char_model, feature_list)
        last_feature = np.array(
            predict_feature_list[0][-1]).reshape(1, 1, 128)
        feature_list = np.append(feature_list, last_feature, axis=1)
        if feature_list.shape[1] >= 20:
            loop = False

    sentence = ""
    for feature in feature_list[0]:
        feature = feature.reshape(1, 4, 4, 8)
        img = Predict.run(decoder_model, feature)
        char = Image2String.image2string(img)
        print(char)
        sentence += char
    print(sentence)


if __name__ == '__main__':
    test()
    # main()
