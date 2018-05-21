

from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
from model.config import Config
data_size = 60000
test_size = 10000


def main():
    train, teach = PreProcessing().cy_make_train_data(data_size, window_size=10)
    test_train, test_teach = PreProcessing().cy_make_train_data(test_size, window_size=25)
    char_model = CharAutoencoder().load_model()
    cbs = CharAutoencoder().set_callbacks(Config.save_model)
    hist = Learning.run_with_test(
        char_model, train, teach, test_train, test_teach, cbs)
    CharAutoencoder().save_model(char_model)


if __name__ == '__main__':
    main()
