class Predict():
    @classmethod
    def run(cls, model, x_test):
        score = model.predict(x_test)
        return score
