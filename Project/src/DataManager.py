import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from Preprocessor import Preprocessor


class DataManager:
    label_col = 'Diagnosis_Age'

    data_frame = pd.DataFrame()
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self, url):
        self.__url = url
        self.__load()
        self.__process_data()
        self.__split_labels_from_data()

    def __load(self):
        if "csv" in self.__url.lower():
            self.data_frame = pd.read_csv(self.__url)
        else:
            self.data_frame = pd.read_table(self.__url, sep='\t')

        # get column name for age column (can be different such as diagnosis_age, age, Age, ...)
        age_column_name = Preprocessor.findAgeColumnName(data_frame=self.data_frame)
        self.label_col = age_column_name
        # remove rows where age == 'nan'
        self.data_frame = Preprocessor.deleteRowIfColumnIsNan(data_frame=self.data_frame, column_name=age_column_name)

        # remove columns where majority of values is Nan
        self.data_frame = Preprocessor.deleteNanColumns(data_frame=self.data_frame, threshold=30)

        # replace nan values with mean of column
        self.data_frame = Preprocessor.replaceNanValuesWithMedian(data_frame=self.data_frame)

        # print(self.data_frame[self.label_col])
        #print(self.data_frame["Diagnosis_Age"])

    def __set_age_groups_as_label(self):
        for i, row in self.data_frame.iterrows():
            if self.data_frame.loc[i][self.label_col] <= 20:
                self.data_frame.at[i, self.label_col] = 0
            elif self.data_frame.loc[i][self.label_col] <= 40:
                self.data_frame.at[i, self.label_col] = 1
            elif self.data_frame.loc[i][self.label_col] <= 60:
                self.data_frame.at[i, self.label_col] = 2
            elif self.data_frame.loc[i][self.label_col] > 60:
                self.data_frame.at[i, self.label_col] = 3

    def __process_data(self):
        self.__set_age_groups_as_label()
        self.__categorical_data_to_numerical()

    def __categorical_data_to_numerical(self):
        dataframe_copy = self.data_frame.select_dtypes(include=['object']).copy()

        # join cols with int and float values
        int_dataframe = self.data_frame.select_dtypes(include=['int64']).copy()
        float_dataframe = self.data_frame.select_dtypes(include=['float64']).copy()
        dataframe_int_float = pd.concat([float_dataframe, int_dataframe], axis=1)

        le = preprocessing.LabelEncoder()
        dataframe_categorical = dataframe_copy.astype(str).apply(le.fit_transform)

        self.data_frame = pd.concat([dataframe_int_float, dataframe_categorical], axis=1)

    def get_whole_dataframe(self):
        return self.data_frame

    def __split_labels_from_data(self):
        dataframe = self.data_frame.copy()
        self.labels = dataframe.pop(self.label_col)
        self.data = dataframe

    def get_labels_and_data(self):
        return self.labels, self.data

