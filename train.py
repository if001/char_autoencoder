
from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np

data_size = 2000


def main():
    train_x, teach_y = PreProcessing().make_train_data(data_size, window_size=10)
    word_autoencoder_model = CharAutoencoder().make_model()
    # cbs = CharAutoencoder().set_callbacks("model.hdf5")
    # hist = Learning.run(word_autoencoder_model, train_x, train_y, cbs)
    exit(0)

    print("test:", train_x[0])
    test_x = np.array([train_x[0]])
    score = Predict.run(word_autoencoder_model, test_x)
    print("score:", score)


if __name__ == '__main__':
    main()
