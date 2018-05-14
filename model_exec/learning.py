import keras

from . import config


class Learning():
    '''
    doc
    '''
    @classmethod
    def run_with_test(cls, model, x_train, y_train, x_test, y_test, cbs):
        history = model.fit(x_train, y_train,
                            batch_size=config.Config.batch_size,
                            epochs=config.Config.epochs,
                            verbose=config.Config.verbose,
                            validation_split=config.Config.validation_split,
                            callbacks=cbs,
                            # validation_data=(x_test, y_test)
                            )
        return history

    @classmethod
    def run(cls, model, x_train, y_train, cbs):
        history = model.fit(x_train, y_train,
                            batch_size=config.Config.batch_size,
                            epochs=config.Config.epochs,
                            callbacks=cbs
                            )
        return history


if __name__ == '__main__':
    pass
