
from model.char_autoencoder import CharAutoencoder
from preprocessing.preprocessing_mod import PreProcessing
from model_exec.learning import Learning
from model_exec.predict import Predict
import numpy as np
from model.config import Config
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
            print("arr len is " + str(len(arr)) + " . " + "unit_name len " +ser(len(cls.unit_name)))
            exit(0)
        cls.depth_dict_arr.append({"prefix":name, "arr":arr})


    @classmethod
    def reshape(cls):
        """
        return dict shape ->
              prefix_unit_name:value,
              arr:value
        """
        __struct_arr = []
        for depth in cls.depth_dict_arr:
            __struct_dict = {}
            for i in range(len(cls.unit_name)):
                __struct_dict["name"] = depth["prefix"] + "_" + cls.unit_name[i]
                __struct_dict["unit"] = depth["arr"][i]
                __struct_arr.append(__struct_dict)
        return __struct_arr


def set_struct():
    Struct.set_unit_name(["min", "small", "large", "big"])
    Struct.depth("shallow"  , [[250],[500],[800],[1200]])
    Struct.depth("middle"   , [[250, 125, 250],[500, 250, 500],[800, 400, 800],[1200, 600, 1200]])
    Struct.depth("middle2"  , [[250, 125, 60, 125, 250],[500,250 ,125, 250, 500],[800, 400, 200, 400, 800],[1200, 600, 300, 600, 1200]])
    Struct.depth("up_middle", [[250, 500, 250, 500, 250],[500,1000, 500, 1000, 500],[800, 1600, 800, 1600, 800],[1200, 2400, 1200, 2400, 1200]])
    Struct.depth("deep"     , [[250, 125, 60, 30, 60, 125, 250],[500, 250, 125, 60, 125, 250, 500],[800, 400, 200, 100, 200, 400, 800],[1200, 600, 300, 150, 300, 600, 1200]])
    Struct.depth("up_deep"  , [[250, 500, 250, 125, 250, 500, 250],[500, 800, 500, 250, 500, 800, 500],[800, 400, 800, 250, 800, 400, 800],[1200, 600, 1200, 300, 1200, 600, 1200]])
    __struct_dict = Struct.reshape()
    return __struct_dict


def main():
    for struct in set_struct():
        char_model = CharAutoencoder().create_model(struct["unit"])
        cbs = CharAutoencoder().set_callbacks(Config.save_model)
        hist = Learning.run(char_model, train, teach, cbs)
        CharAutoencoder().save_model(char_model, struct["name"])


if __name__ == '__main__':
    main()
