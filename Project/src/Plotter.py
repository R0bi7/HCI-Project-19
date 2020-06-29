import shap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=["0-18", "19-50", "50+"], index=["0-18", "19-50", "50+"])
    sns.heatmap(cm,annot=True,cmap='Blues', fmt='g')
    plt.show()


def plot_feature_importance_for_class(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    shap.summary_plot(shap_values[0], X_train)
    shap.summary_plot(shap_values[1], X_train)
    shap.summary_plot(shap_values[2], X_train)


def plot_prediction_desicion(model, X_test, pred, row_idx):
    #The decision plot below shows the model’s multiple outputs for a single observation
    #the dashed line is the prediction of our classifier
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.multioutput_decision_plot([0, 1, 2], shap_values,
                                   row_index=row_idx,
                                   feature_names=list(X_test.columns) ,
                                   highlight=int(pred[row_idx]),
                                   legend_labels=["0-18", "19-50", "50+"],
                                   legend_location='lower right')


def plot_data_balance(data_frame, label_col):
    data_frame.groupby(label_col).Gender.count().plot.bar(ylim=0)
    plt.show()
