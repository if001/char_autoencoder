
from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np

data_size = 100


def main():
    train, teach = PreProcessing().make_train_data(data_size, window_size=10)
    char_autoencoder_model = CharAutoencoder().make_simple_model()
    cbs = CharAutoencoder().set_callbacks("char_model.hdf5")
    hist = Learning.run(char_autoencoder_model, train, teach, cbs)

    # print("test:", train_x[0])
    # test_x = np.array([train_x[0]])
    # score = Predict.run(word_autoencoder_model, test_x)
    # print("score:", score)


if __name__ == '__main__':
    main()
