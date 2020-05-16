import pandas as pd
import Plotter as Plotter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class Classifier:
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self, labels, data):
        self.data = data
        self.labels = labels

    def predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, random_state=0)

        # Stack overflow:
        # TODO You MUST first split into train and test, then replace NA by mean on train and then apply this stateful preprocessing model to test




        model = RandomForestClassifier(max_depth=7, random_state=4)
        model.fit(X_train, y_train)

        # Plotter.plot_feature_importance_for_class(model, X_train)

        y_pred = model.predict(X_test)
        # Plotter.plot_confusion_matrix(y_test, y_pred)
        # Plotter.plot_prediction_desicion(model, X_test, y_pred, 0)

        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        print("Accuracy = " + str(round(accuracy_score(y_test, y_pred) * 100, 2)) + " %")





