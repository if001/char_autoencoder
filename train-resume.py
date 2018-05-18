

from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
from model.config import Config
data_size = 100


def main():
    train, teach = PreProcessing().make_train_data(data_size, window_size=10)
    char_model = CharAutoencoder().load_model()
    cbs = CharAutoencoder().set_callbacks(Config.save_model)
    hist = Learning.run(char_model, train, teach, cbs)
    CharAutoencoder().save_model(char_model)

    # train_x, train_y = PreProcessing().make_train_data()
    # mnist_model = Mnist().load_model()
    # hist = Learning.run(mnist_model, train_x, train_y)

    # test_x, test_y = PreProcessing().make_test_data()
    # score = Predict.run(mnist_model, test_x, test_y)

    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
