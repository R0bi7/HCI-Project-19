from src.Classifier import Classifier
from src.DataManager import DataManager
import glob


def main():
    # get list of url for each csv data in Project->data directory
    csv_files = glob.glob('../data/*.csv')

    # iterate through data files and make model, plot and evaluate
    for url in csv_files:
        datamanager = DataManager(url=url)
        labels, data = datamanager.get_labels_and_data()
        classfier = Classifier(labels, data)
        classfier.predict()


if __name__ == "__main__":
    main()
