import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.random import seed


class AdalineSGD(object):
    """
    ADAptive LInear NEuron classifier
    
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset
        
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch
    shuffle : bool(Default = true)
        Shuffles training data every epoch
        if True to precent cycles.
    random_state : int (default : None)
        Set random state for shuffling
        and initializing the weights
    """

    def __init__(self, eta=0.01, n_iter=0,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if (random_state):
            seed(random_state)
        self.f = open('adalineSGDOutput.txt', 'w+')

    def fit(self, X, y):
        """
        Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Trainging vectors, where n_samples is the 
            number of samples and n_features is the number of features.
        Y : array-like, shape = [n_samples]
            Target values.
            
        Return
        ------
        self : object
        
        :param X: 
        :param y: 
        :return: 
        """
        s = "\nX = " + str(X)
        self.f.write(s)

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            s = "\n\n\n------------------------------------\n\nCurrent iteration = " + str(i)
            self.f.write(s)

            cost = []

            for xi, target in zip(X, y):
                current_cost = self._update_weights(xi, target)
                cost.append(current_cost)
                s = "\n self._update_weights(xi, target) = " + str(current_cost)
                self.f.write(s)
            avg_cost = sum(cost) / len(y)
            s = "   average cost = " + str(avg_cost)
            self.f.write(s)
            self.cost_.append(avg_cost)
        return self

    # def partial_fit(self, X, y):
    #     """ Fit training data without reinitializing the weights """
    #     if not self.w_initialized:
    #         self._initialize_weights(X.shape[1])
    #     if y.ravel().shape[0] > 1:
    #         s = "\n\n\ny.ravel() = " + str(y.ravel())
    #         self.f.write(s)
    #         s = "\n\n\n y.ravel().shape[0] = " + str(y.ravel().shape[0])
    #         self.f.write(s)
    #         for xi, target in zip(X, y):
    #             self._update_weights(xi, target)
    #     else:
    #         self._update_weights(X, y)
    #     return self

    def _shuffle(self, X, y):
        """ Shuffle training data """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ Initialize wieghts to zeros """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ Apply Adaline learning rule to the weights """
        output = self.net_input(xi)
        s = "    output = " + str(output)
        self.f.write(s)
        s = "    target = " + str(target)
        self.f.write(s)
        error = (target - output)
        s = "    error one = " + str(error)
        self.f.write(s)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate the net input"""
        print("str(np.dot(X, self.w_[1:])) = " + str(np.dot(X, self.w_[1:])))
        print("self.w_[0]" + str(self.w_[0]))
        print(str(np.dot(X, self.w_[1:]) + self.w_[0]))
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
    # plt.scatter(X4[0:50, 0], X4[0:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X4[50:100, 0], X4[50:100, 1],
    #             color='blue', marker='x', label='versicolor')
    # plt.xlabel('petal length')
    # plt.ylabel('sepal length')
    # plt.legend(loc='upper left')
    # plt.show()
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # ada1 = AdalineSGD(n_iter=10, eta=0.01).fit(X4, y)
    # ax[0].plot(range(1, len(ada1.cost_) + 1),
    #            np.log10(ada1.cost_), marker='o')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('log(Sum-squared-erros)')
    # ax[0].set_title('Adaline - Learning rate 0.01')
    # ada2 = AdalineSGD(n_iter=10, eta=0.0001).fit(X4, y)
    # ax[1].plot(range(1, len(ada2.cost_) + 1),
    #            np.log10(ada2.cost_), marker='o')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('log(Sum-squared-erros)')
    # ax[1].set_title('Adaline - Learning rate 0.0001')
    # plt.show()

    X_std = np.copy(X4)
    X_std[:, 0] = (X4[:, 0] - X4[:, 0].mean()) / X4[:, 0].std()
    X_std[:, 1] = (X4[:, 1] - X4[:, 1].mean()) / X4[:, 1].std()

    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized')
    plt.ylabel('pedal length [standardized')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average cost')
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


analyze()
