
from model.mnist import Mnist
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict

def main():
    train_x, train_y = PreProcessing().make_train_data()
    mnist_model = Mnist().load_model()
    hist = Learning.run(mnist_model, train_x, train_y)

    test_x, test_y = PreProcessing().make_test_data()
    score = Predict.run(mnist_model, test_x, test_y)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
   main()
