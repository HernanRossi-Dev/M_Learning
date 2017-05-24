import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class AdaptiveLinearNeuron(object):

    def __init__(self, eta=0.01, n_iter=0):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self, X, y):
        f = open('adalineOutput.txt', 'w+')
        s = "\nX = " + str(X)
        f.write(s)
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            s = "\n\n\nCurrent iteration = " + str( i)
            f.write(s)
            s = "\n\nw_[1:] = " + str(self.w_[1:])
            f.write(s)
            output = self.net_input(X)
            s = "\nnp.dot(X, self.w_[1:]) = " + str( np.dot(X, self.w_[1:]))
            f.write(s)
            s = "\nself.w_[0] = " + str( self.w_[0])
            f.write(s)
            s = "\noutput = ( np.dot(X, self.w_[1:]) + self.w_[0]) = " + str(output)
            f.write(s)
            errors = (y - output)
            s = "\nerrors = " + str(errors.T)
            f.write(s)
            self.w_[1:]+= self.eta * X.T.dot(errors)
            s = "\nself.w_ = " +  str(self.w_)
            f.write(s)
            s = "\nerrors.sum() = " + str(errors.sum())
            f.write(s)
            self.w_[0] +=self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            s = str(cost)
            f.write(s)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute Linear Activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

def analyze():
    df = pd.read_csv('Iris.csv', header=None)
    print(df.tail())
    y = df.iloc[1:101, 5].values
    print(y)
    y = np.where(y == 'Iris-setosa', -1, 1)
    print(y)
    X4 = df.iloc[1:101, [1, 3]].values
    X2 = np.array(X4[0:100, 0], dtype=float).T
    X3 = np.array(X4[0:100, 1], dtype=float).T
    X4 = np.vstack((X2, X3)).T
    print(X4)
    plt.scatter(X4[0:50, 0], X4[0:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X4[50:100, 0], X4[50:100, 1],
                color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    ada1 = AdaptiveLinearNeuron(n_iter=10, eta=0.01).fit(X4 , y)
    ax[0].plot(range(1, len(ada1.cost_) + 1),
               np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-erros)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdaptiveLinearNeuron(n_iter=10, eta=0.0001).fit(X4,y)
    ax[1].plot(range(1, len(ada2.cost_) + 1),
               np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-erros)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

    X_std = np.copy(X4)
    X_std[:,0] = (X4[:,0] - X4[:,0].mean()) / X4[:,0].std()
    X_std[:, 1] = (X4[:, 1] - X4[:, 1].mean()) / X4[:, 1].std()

    ada = AdaptiveLinearNeuron(n_iter= 15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized')
    plt.ylabel('pedal length [standardized')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum of squared-errors')
    plt.show()



def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

analyze()