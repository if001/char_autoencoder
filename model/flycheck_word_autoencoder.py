from . import abc_model
from . import config

import keras
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers.wrappers import TimeDistributed as TD
from keras.models import Model


class WordAutoencoder(abc_model.ABCModel):
    @classmethod
    def set_callbacks(cls, fname):
        # fname = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
        fpath = config.Config.run_dir_path + "/weight/" + fname
        callbacks = []
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto'))

        return callbacks

    @classmethod
    def make_model(cls):
        input_layer = Input(shape=(None, 4 * 4 * 8))
        output_layer = Dense(512, activation='relu')(input_layer)
        output_layer, state_h, state_c = LSTM(
            256, return_state=True, return_sequences=True)(output_layer)
        output_layer = Dense(512, activation='relu')(output_layer)
        output_layer = Dense(128, activation='relu')(output_layer)
        model = Model(input_layer, output_layer)
        model.summary()

        model.compile(loss=config.Config.loss,
                      optimizer=config.Config.optimizer,
                      metrics=[config.Config.metrics])
        return model

    @classmethod
    def save_model(cls, model):
        print("save" + config.Config.save_model)
        model.save(config.Config.save_model)

    @classmethod
    def load_model(cls):
        print("load" + config.Config.save_model)
        from keras.models import load_model
        return load_model(config.Config.save_model)
