import shap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=[0, 1, 2, 3], index=[0, 1, 2, 3])
    sns.heatmap(cm, annot=True)
    plt.show()


def plot_feature_importance_for_class(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")