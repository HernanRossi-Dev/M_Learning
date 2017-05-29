import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class TestDataPrep:
    def __init__(self):
        return

    def prepareTestData(self):
        df_wine = pd.read_csv('wine.data', header=None)
        df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                           'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                           'Hue', 'OD280/OD315 of diluted wines', 'Proline']
        print('Class labels', np.unique(df_wine['Class label']))
        X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        mms = MinMaxScaler()
        # normalized data on range 0,1
        X_train_norm = mms.fit_transform(X_train)
        X_test_norm = mms.transform(X_test)
        stdsc = StandardScaler()
        X_train_std = stdsc.fit_transform(X_train)
        X_test_std = stdsc.transform(X_test)
        df_wine.head()


def run():
    test = TestDataPrep()
    test.prepareTestData()


run()
