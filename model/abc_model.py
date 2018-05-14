from abc import ABCMeta, abstractmethod

class ABCModel():
    @abstractmethod
    def make_model():
        pass
    @abstractmethod
    def save_model():
        pass
    @abstractmethod
    def load_model():
        pass
