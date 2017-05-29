import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer


class TestDataFrame:
    def __init__(self, csv_data):
        self.csv_data = csv_data

    def createTestData(self):
        df = pd.read_csv(StringIO(self.csv_data))
        print(df)
        print(df.isnull().sum())
        print(df.values)
        print(df.dropna())
        print(df.dropna(axis=1))
        #only drop rows where all columns are NaN
        print(df.dropna(how='all'))
        print()
        # drop rows that have at least 4 non-Nan values
        print(df.dropna(thresh=4))
        print()
        # only drop rows where Nan appear in specific columns(here: 'C')
        print(df.dropna(subset=['C']))
        print()

        # Interpolation using mean imputation to replace missing feature values
        imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imr = imr.fit(df)
        imputed_data = imr.transform(df.values)
        print(imputed_data)


def run():
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0,'''

    test = TestDataFrame(csv_data)
    test.createTestData()


run()
