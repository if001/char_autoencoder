from preprocessing.preprocessing_mod import PreProcessing

from model.config import Config
from model.char_autoencoder import CharAutoencoder
from model_exec.learning import Learning
from keras.backend import tensorflow_backend as backend
import numpy as np

from keras.backend import tensorflow_backend as backend
from keras import backend as K
from keras.optimizers import RMSprop, Adam, Adadelta, SGD

data_size = 60000
test_size = 10000


class Struct():
    unit_name = []
    depth_dict_arr = []

    @classmethod
    def set_unit_name(cls, unit_name):
        cls.unit_name = unit_name

    @classmethod
    def depth(cls, name, arr):
        if len(arr) != len(cls.unit_name):
            print("arr len is " + str(len(arr)) + " . " +
                  "unit_name len " + ser(len(cls.unit_name)))
            exit(0)
        cls.depth_dict_arr.append({"prefix": name, "arr": arr})

    @classmethod
    def reshape(cls):
        """
        return dict shape ->
              prefix_unit_name:value,
              arr:value
        """
        __struct_arr = []
        for depth in cls.depth_dict_arr:
            for i in range(len(cls.unit_name)):
                __struct_dict = {}
                __struct_dict["name"] = depth["prefix"] + \
                    "_" + cls.unit_name[i]
                __struct_dict["unit"] = depth["arr"][i]
                __struct_arr.append(__struct_dict)
        return __struct_arr


def set_struct():
    Struct.set_unit_name(["min", "small", "large", "big"])
    Struct.depth("shallow", [[64], [128], [256], [512]])
    Struct.depth("middle", [[64, 32, 64], [128, 64, 128], [
                 256, 128, 256], [512, 256, 512]])
    Struct.depth("middle2", [[64, 32, 16, 32, 64], [128, 64, 32, 64, 128], [
                 256, 128, 64, 128, 256], [512, 256, 128, 256, 512]])
    Struct.depth("up_middle", [[8, 16, 8, 16, 8], [16, 32, 16, 32, 16], [
                 32, 64, 32, 64, 32], [64, 128, 64, 128, 64]])
    # Struct.depth("deep", [[250, 125, 60, 30, 60, 125, 250], [500, 250, 125, 60, 125, 250, 500], [
    #              800, 400, 200, 100, 200, 400, 800], [1200, 600, 300, 150, 300, 600, 1200]])
    # Struct.depth("up_deep", [[250, 500, 250, 125, 250, 500, 250], [500, 800, 500, 250, 500, 800, 500], [
    #              800, 400, 800, 250, 800, 400, 800], [1200, 600, 1200, 300, 1200, 600, 1200]])
    __struct_dict = Struct.reshape()

    # __struct_dict = [
    #     {
    #         "name": "min_shallow",
    #         "unit": [800]
    #     },
    #     {"name": "small_shallow",
    #      "unit": [500, 250, 125, 250, 500]
    #      },
    #     {
    #         "name": "large_shallow",
    #         "unit": [800, 400, 800]
    #     },
    #     {
    #         "name": "big_shallow",
    #         "unit": [1200, 600, 1200]
    #     }
    # ]
    return __struct_dict


def main():
    train, teach = PreProcessing().load_train_data()

    for struct in set_struct():
        for i in range(3):
            char_model = CharAutoencoder.create_model(struct["unit"], i)
            cbs = CharAutoencoder.set_callbacks(struct["name"])
            hist = Learning.run(char_model, train, teach, cbs)
            CharAutoencoder.save_model(char_model, struct["name"])
            CharAutoencoder.clear_session()
            backend.clear_session()
        # K.clear_session()


if __name__ == '__main__':
    main()
