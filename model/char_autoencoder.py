from . import abc_model
from . import config

import keras
from keras.layers import Dense, Dropout, Input, LSTM
from keras.layers.wrappers import TimeDistributed as TD
from keras.models import Model


class CharAutoencoder(abc_model.ABCModel):
    @classmethod
    def set_callbacks(cls, fname):
        # fname = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
        # fpath = config.Config.run_dir_path + "/weight/" + fname
        # print(fpath)
        callbacks = []
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=fname, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto'))

        return callbacks

    @classmethod
    def make_simple_model(cls):
        layer_input = Input(shape=(None, 4 * 4 * 8))
        x = Dense(256, activation='sigmoid')(layer_input)
        x = LSTM(512, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = LSTM(512, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        layer_output = Dense(128, activation='relu')(x)
        model = Model(layer_input, layer_output)
        model.summary()
        model.compile(loss=config.Config.loss,
                      optimizer=config.Config.optimizer,
                      metrics=[config.Config.metrics])
        return model

    @classmethod
    def make_model(cls):
        encoder_input = Input(shape=(None, 4 * 4 * 8))

        x = Dense(512, activation='sigmoid')(encoder_input)
        x, state_h, state_c = LSTM(
            256, return_state=True)(x)
        states = [state_h, state_c]
        decoder_input = Input(shape=(None, 4 * 4 * 8))
        x = Dense(512, activation='sigmoid')(decoder_input)
        decoder_output = LSTM(256, return_sequences=True)(
            x, initial_state=states)

        model = Model([encoder_input, decoder_input], decoder_output)
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
