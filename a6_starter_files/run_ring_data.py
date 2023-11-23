"""run_2_class_2_feature_iris_data.py
YOU SHOULD NOT NEED TO EDIT THIS FILE OR TURN IT IN.
HOWEVER, YOU ARE WELCOME TO EDIT THE FILE TO EXPLORE
POSSIBLE ADJUSTMENTS TO PARAMETERS.

Train a perceptron on the first 10 examples of iris setosa
and the first 10 examples of iris versicolor, considering
only sepal length and petal length as features.

Then test with the remaining 40 examples of each.
Extends the Class PlotBinaryPerceptron

Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.
"""

from binary_perceptron import BinaryPerceptron # Your implementation of binary perceptron
from plot_bp import PlotBinaryPerceptron
import csv  # For loading data.
from matplotlib import pyplot as plt  # For creating plots.
from remapper import remap


class PlotRingBP(PlotBinaryPerceptron):
    """
    Plots the Binary Perceptron after training it on the Iris dataset
    ---
    Extends the class PlotBinaryPerceptron
    """

    def __init__(self, bp, plot_all=True, n_epochs=25, is_remapped=False):
        super().__init__(bp, plot_all, n_epochs) # Calls the constructor of the super class
        self.is_remapped = True


    def read_data(self):
        """
        Read data from the Iris dataset with 2 features and 2 classes
        for both training and testing.
        ---
        Overrides the method in PlotBinaryPerceptron
        """
        data_as_strings = list(csv.reader(open('ring-data.csv'), delimiter=','))
        if not self.is_remapped:
            self.TRAINING_DATA = [[float(f1), float(f2), int(c)] for [f1, f2, c] in data_as_strings]
        else:
            # get training data 
            data = []
            for [f1, f2, c] in data_as_strings:
                remapped_point = remap(float(f1), float(f2))
                data.append([remapped_point[1], remapped_point[0], int(c)])
            self.TRAINING_DATA = data

    def plot(self):
        """
        Plots the dataset as well as the binary classifier
        ---
        Overrides the method in PlotBinaryPreceptron
        """
        plt.title("Ring dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(loc='best')
        plt.show()



if __name__=='__main__':
    binary_perceptron = BinaryPerceptron(alpha=0.5)
    pbp = PlotRingBP(binary_perceptron)
    pbp.train()
    pbp.plot()
    
