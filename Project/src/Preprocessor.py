import pandas as pd


class Preprocessor:
    @staticmethod
    def deleteRowIfColumnIsNan(data_frame: pd.DataFrame, column_name: str):
        for index, row in data_frame.iterrows():
            if not (row[column_name] > 0):
                data_frame.drop(index, inplace=True)
        return data_frame

    @staticmethod
    def replaceNanValuesWithMedian(data_frame: pd.DataFrame):
        data_frame.fillna(data_frame.median(), inplace=True)
        return data_frame
