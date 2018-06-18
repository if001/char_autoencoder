from preprocessing.preprocessing_mod import PreProcessing

from model.config import Config
from model.char_autoencoder import CharAutoencoder
from model_exec.learning import Learning
from keras.backend import tensorflow_backend as backend
from keras.backend import tensorflow_backend as backend
from keras import backend as K
from keras.optimizers import RMSprop, Adam, Adadelta, SGD
import numpy as np
import os
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
                  "unit_name len " + str(len(cls.unit_name)))
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

    __struct_dict = [
        {"name": "small_shallow",
         "unit": [512, 256, 512]
         }
    ]
    return __struct_dict


def main():
    hists = []
    opt = Adam

    lrs = np.arange(1, 11, 1) / 10000
    lrs = [0.0002]

    prefix_list = []
    for f in os.listdir("./preprocessing/"):
        if ".npz" in f:
            prefix_list.append("-".join(f.split("-")[1:]))
    prefix_list = list(set(prefix_list))

    for struct in set_struct():
        for lr in lrs:
            print(lr)
            tmp_hists = []
            char_model = CharAutoencoder.create_model(
                struct["unit"], Adam, lr)
            cbs = CharAutoencoder.set_callbacks(struct["name"])
            for fname in prefix_list:
                print("load", fname)
                train, teach = PreProcessing().load_split_train_data(fname)
                hist = Learning.run(char_model, train, teach, cbs)
                print(hist)
                CharAutoencoder.save_model(char_model, struct["name"])
            CharAutoencoder.clear_session()
            backend.clear_session()

            tmp_hists.append(struct["unit"])
            opt_name = str(opt()).split(" ")[0].split(".")[-1]
            tmp_hists.append({"optimizer": opt_name})
            tmp_hists.append({"lr": str(lr)})
            tmp_hists.append(hist.history)
            print(tmp_hists)
        hists.append(tmp_hists)

    print(hists)


if __name__ == '__main__':
    main()
