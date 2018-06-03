from . import abc_model
from . import config

import keras
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers.wrappers import TimeDistributed as TD
from keras.models import Model
from keras.backend import tensorflow_backend as backend
from keras.optimizers import RMSprop, Adam, Adadelta, SGD


class CharAutoencoder(abc_model.ABCModel):
    @classmethod
    def clear_session(cls):
        backend.clear_session()

    @classmethod
    def set_callbacks(cls, fname):
        # fpath = config.Config.run_dir_path + "/weight/" + fname
        # print(fpath)
        fname = config.Config.run_dir_path + "/weight/" + "char_model-" + fname + ".hdf5"
        callbacks = []
        # callbacks.append(keras.callbacks.ModelCheckpoint(
        #     filepath=fname, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=0, verbose=1, mode='auto'))

        # callbacks.append(keras.callbacks.TensorBoard(
        #     log_dir=config.Config.run_dir_path + '/tflog', histogram_freq=1))

        return callbacks

    @classmethod
    def make_simple_model(cls):
        layer_input = Input(shape=(None, 4 * 4 * 8))
        x = LSTM(800, return_sequences=True)(layer_input)
        x = Dropout(0.3)(x)
        x = LSTM(500, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = LSTM(500, return_sequences=True)(x)
        x = Dropout(0.7)(x)
        x = LSTM(800, return_sequences=True)(x)
        x = Dropout(0.7)(x)
        layer_output = Dense(128, activation='relu')(x)
        model = Model(layer_input, layer_output)
        model.summary()
        model.compile(loss=config.Config.loss,
                      optimizer=config.Config.optimizer,
                      metrics=[config.Config.metrics])
        return model

    # decoratorとか作ったけどいらんかった
    # けど頑張ったから残す
    # def make_model(func):
    #     def _new_func(*args, **kwargs):
    #         model = func(*args, **kwargs)
    #         model.summary()
    #         model.compile(loss=config.Config.loss,
    #                        optimizer=config.Config.optimizer,
    #                        metrics=[config.Config.metrics])
    #         return model
    #     return _new_func

    # # run for colaboratory
    # @make_model
    # def __model(cls, struct):
    #     layer_input = Input(shape=(None, 4 * 4 * 8))
    #     _in = layer_input
    #     for i in range(len(struct)):
    #         x = LSTM(struct[i], return_sequences=True)(layer_input)
    #         x = Dropout(0.5)(x)
    #     layer_output = Dense(128, activation='relu')(x)
    #     model = Model(layer_input, layer_output)
    #     return model

    # @classmethod
    # def create_model(cls, struct):
    #     return cls.__model(cls, struct)

    @classmethod
    def create_model(cls, struct, idx):
        opts = [Adadelta(), RMSprop(), Adam(), SGD()]
        layer_input = Input(shape=(None, 4 * 4 * 8))
        __in = layer_input
        for i in range(len(struct)):
            x = LSTM(struct[i], return_sequences=True)(__in)
            x = Dropout(0.5)(x)
            __in = x

        layer_output = Dense(128, activation='relu')(x)
        model = Model(layer_input, layer_output)
        model.summary()
        model.compile(loss=config.Config.loss,
                      optimizer=opts[i],
                      metrics=[config.Config.metrics])
        return model

    @classmethod
    def save_model(cls, model, name):
        fname = config.Config.run_dir_path + "/weight/" + "char_model-" + name + ".hdf5"
        print("save" + fname)
        model.save(fname)

    @classmethod
    def load_model(cls):
        print("load" + config.Config.save_model)
        from keras.models import load_model
        return load_model(config.Config.save_model)
