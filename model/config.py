from keras.optimizers import RMSprop, Adam
import os


class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/char_model.hdf5"
    loss = 'binary_crossentropy'
    loss = 'mean_squared_error'
    optimizer = 'adam'
    optimizer = Adam(lr=0.0007)
    metrics = 'accuracy'
