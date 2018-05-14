from abc import ABCMeta, abstractmethod
class ABCPreProcessing(metaclass=ABCMeta):
    @abstractmethod
    def make_train_data(self):
        pass

    @abstractmethod
    def make_test_data(self):
        pass
