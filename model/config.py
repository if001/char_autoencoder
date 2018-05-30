from keras.optimizers import RMSprop, Adam, Adadelta,SGD
import os


class Config():
    run_dir_path = os.path.dirname(os.path.abspath(__file__))
    save_model = run_dir_path + "/weight/char_model.hdf5"
    loss = 'binary_crossentropy'
    loss = 'mean_squared_error'
    # print(0.0005)
    # optimizer = Adam(lr=0.0005,decay=0.001)
    #optimizer = Adam(lr=0.0008)
    # optimizer = Adadelta()
    # optimizer = SGD()
    metrics = 'accuracy'
