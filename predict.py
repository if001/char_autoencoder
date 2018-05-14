
from model.mnist import Mnist
from preprocessing.preprocessing import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict

def main():
    mnist_model = Mnist().load_model()
    test_x, test_y = PreProcessing().make_test_data()
    score = Predict.run(mnist_model, test_x, test_y)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
   main()
