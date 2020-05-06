import pandas as pd

from sklearn.model_selection import train_test_split


class Classifier:
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self, labels, data):
        self.data = data
        self.labels = labels

    def predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, random_state=0)




