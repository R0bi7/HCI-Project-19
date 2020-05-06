import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.Classifier import Classifier
from src.DataManager import DataManager

categorical_cols = ['Cancer_Type', 'Cancer_Type_Detailed', 'Oncotree_Code', 'Ethnicity_Category', 'Race_Category', 'Sex']

numerical_cols = ['Fraction_Genome_Altered', 'Mutation_Count', 'Neoplasm_Histologic_Grade']

label_col = 'Diagnosis_Age'

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def set_age_groups_as_label(dataframe):
    for i, row in dataframe.iterrows():
        if dataframe.loc[i][label_col] > 20:
            dataframe.at[i, label_col] = 0
        elif dataframe.loc[i][label_col] > 40:
            dataframe.at[i, label_col] = 1
        elif dataframe.loc[i][label_col] > 60:
            dataframe.at[i, label_col] = 2
        elif dataframe.loc[i][label_col] > 80:
            dataframe.at[i, label_col] = 3


    return dataframe


def load_dataframe():
    URL = '../data/data2.csv'
    dataframe = pd.read_csv(URL)
    dataframe.head()
    dataframe = dataframe.dropna()
    dataframe = set_age_groups_as_label(dataframe)

    return dataframe


def create_train_val_test_sets(dataframe):
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    return train, test, val


def predict(train_ds, val_ds, test_ds, feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        #layers.Dense(1)
    ])

    #Could probably be used for multiclass classification instead of the model.compile below which is only for binary classification
    model.compile(
        optimizer='adam',
        loss=['sparse_categorical_crossentropy'],
        metrics=['accuracy']
    )

    #model.compile(optimizer='adam',
    #              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #              metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy, "Loss", loss)


def create_feature_columns(dataframe):
    feature_columns = []

    # numeric cols
    for header in numerical_cols:
        feature_columns.append(feature_column.numeric_column(header))

    # categorical cols
    for colHeader in categorical_cols:
        column = feature_column.categorical_column_with_vocabulary_list(
            colHeader, dataframe[colHeader].unique())
        one_hot = feature_column.indicator_column(column)
        feature_columns.append(one_hot)

    return feature_columns


def main():
    datamanager = DataManager()
    labels, data = datamanager.get_labels_and_data()
    classfier = Classifier(labels, data)
    classfier.predict()
    #dataframe = load_dataframe()

    #train, test, val = create_train_val_test_sets(dataframe)

    #feature_columns = create_feature_columns(dataframe)
    #feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    #batch_size = 32
    #train_ds = df_to_dataset(train, batch_size=batch_size)
    #val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    #test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # for feature_batch, label_batch in train_ds.take(1):
    #    print('Every feature:', list(feature_batch.keys()))
    #    print('A batch of Cancer Types:', feature_batch['Cancer Type'])
    #    print('A batch of targets:', label_batch)

    #predict(train_ds, val_ds, test_ds, feature_layer)


if __name__ == "__main__":
    main()
