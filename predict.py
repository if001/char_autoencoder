

from model.char_autoencoder import CharAutoencoder

from preprocessing.preprocessing import PreProcessing
from model_exec.predict import Predict
import numpy as np

import sys
sys.path.append("../")
# from string2image.image2string import image2string
from string2image.image2string import Image2String
from cnn_autoencoder.model.simple_autoencoder import SimpleAutoencoder
# from cnn_autoencoder.model_exec.predict import Predict


def main():
    char_model = CharAutoencoder().load_model()

    feature_list = PreProcessing().make_test_data("あ")
    loop = True
    while(loop):
        predict_feature_list = Predict.run(char_model, feature_list)
        last_feature = np.array(
            predict_feature_list[0][-1]).reshape(1, 1, 128)
        feature_list = np.append(feature_list, last_feature, axis=1)
        if feature_list.shape[1] == 5:
            loop = False

    cnn_model = SimpleAutoencoder().load_model("cnn_model.hdf5")
    decoder_model = SimpleAutoencoder().make_decoder_model(cnn_model)

    sentence = ""

    for feature in feature_list[0]:
        feature = feature.reshape(1, 4, 4, 8)
        img = Predict.run(decoder_model, feature)
        char = Image2String.image2string(img)
        print(char)
        sentence += char
    print(sentence)


if __name__ == '__main__':
    main()
