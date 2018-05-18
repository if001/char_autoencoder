
from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np
from model.config import Config
data_size = 100


def main():
    train, teach = PreProcessing().make_train_data(data_size, window_size=10)
    char_model = CharAutoencoder().make_simple_model()
    cbs = CharAutoencoder().set_callbacks(Config.save_model)
    hist = Learning.run(char_model, train, teach, cbs)
    CharAutoencoder().save_model(char_model)


if __name__ == '__main__':
    main()
