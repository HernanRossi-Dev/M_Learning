import numpy as np
import matplotlib.pyplot as plt
import itertools as itools


class UniformScatter:
    """ class to create a uniform scatter plot """

    def __init__(self, numPoints):
        self.numPoints = numPoints

    def createPlot(self):
        xValues = np.arange(0, self.numPoints, 25)
        yValues = np.arange(0, self.numPoints, 35)
        xAxis = []
        yAxis =[]
        for x , y in itools.product(xValues, yValues):
            xAxis.append(x)
            yAxis.append(y)


        fig = plt.figure()
        ax=fig.gca()
        ax.set_xticks(np.arange(0, 1000, 20))
        ax.set_yticks(np.arange(0, 1000, 50))
        plt.scatter(xAxis, yAxis, alpha=0.8, linewidths=0.5, c='b', marker='x')
        plt.grid()
        plt.show()


def run():
    test = UniformScatter(1000)
    test.createPlot()


run()
