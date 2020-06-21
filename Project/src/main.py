from src.Classifier import Classifier
from src.DataManager import DataManager


def main():
    datamanager = DataManager()
    labels, data = datamanager.get_labels_and_data()
    classfier = Classifier(labels, data)
    classfier.predict()


if __name__ == "__main__":
    main()
