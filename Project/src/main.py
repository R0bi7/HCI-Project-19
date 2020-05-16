from Classifier import Classifier
from DataManager import DataManager
import glob
import traceback


def main():
    # get list of url for each csv data in Project->data directory
    # csv_files = glob.glob('../data/*.csv')
    csv_files = glob.glob('../robi-data/*.tsv')

    # iterate through data files and make model, plot and evaluate
    for url in csv_files:
        try:
            print(url)
            datamanager = DataManager(url=url)
            labels, data = datamanager.get_labels_and_data()
            classfier = Classifier(labels, data)
            classfier.predict()
        except:
            print("Exception for " + str(url))
            pass


if __name__ == "__main__":
    main()
