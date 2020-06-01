import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataManager:
    label_col = 'Diagnosis_Age'

    __mutations = pd.DataFrame()
    __ageInfo = pd.DataFrame()

    data_frame = pd.DataFrame()
    data = pd.DataFrame()
    labels = pd.DataFrame()

    def __init__(self):
        self.__load()
        self.__process_data()
        self.__split_labels_from_data()

    def __load(self):
        self.__ageInfo = pd.read_csv("../data/mutation_age_info.csv")
        self.__mutations = pd.read_csv("../data/mutations.txt", sep='\t')

    def __set_age_groups_as_label(self):
        for i, row in self.data_frame.iterrows():
            if self.data_frame.loc[i][self.label_col] <= 18:
                self.data_frame.at[i, self.label_col] = 0
            elif self.data_frame.loc[i][self.label_col] <= 50:
                self.data_frame.at[i, self.label_col] = 1
            elif self.data_frame.loc[i][self.label_col] > 50:
                self.data_frame.at[i, self.label_col] = 2

    def __process_data(self):
        self.__set_text_as_true_and_nan_as_false()
        self.__merge_frames()
        self.__set_age_groups_as_label()
        print(self.data_frame)
        #self.__categorical_data_to_numerical()

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

    def __merge_frames(self):
        self.__ageInfo = self.__ageInfo.drop_duplicates(subset=['Sample_ID'], keep='first')
        self.data_frame = pd.merge(self.__mutations, self.__ageInfo[["Diagnosis_Age", "Sample_ID", "Mutation_Count"]], on="Sample_ID")
        self.data_frame = self.data_frame.drop(columns=['Sample_ID', 'STUDY_ID'])
        self.data_frame = self.data_frame.dropna()

    def __set_text_as_true_and_nan_as_false(self):
        sampleid = self.__mutations[['Sample_ID', "STUDY_ID"]].copy()
        self.__mutations = self.__mutations.drop(columns=['Sample_ID', 'STUDY_ID'])
        self.__mutations = pd.DataFrame(np.where(self.__mutations.isna(), self.__mutations, 1), columns=self.__mutations.columns)
        self.__mutations = self.__mutations.fillna(0)
        self.__mutations = self.__mutations.join(sampleid)
