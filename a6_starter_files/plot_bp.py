"""plot_bp.py
YOU SHOULD NOT NEED TO EDIT THIS FILE OR TURN IT IN.
HOWEVER, YOU ARE WELCOME TO EDIT THE FILE TO EXPLORE
POSSIBLE ADJUSTMENTS TO PARAMETERS.

Implements a class that can train and plot a binary perceptron
for any dataset with 2 features, and classes +1 and -1.

Version 1.1, Prashant Rangarajan and S. Tanimoto, May 11, 2021. Univ. of Washington.
"""

from binary_perceptron import BinaryPerceptron  # Your implementation of binary perceptron
from matplotlib import pyplot as plt  # For creating plots.


class PlotBinaryPerceptron:
    """
    Plots the Binary Perceptron after training it on a dataset
    with classes +1 and -1
    """
    
    def __init__(self, bp, plot_all=True, n_epochs=10):
        """
        Initializes the class
        ---
        X_MIN: Minimum X coordinate of the data for the plot
        X_MAX: Maximum X coordinate of the data for the plot
        TRAINING_DATA: To be filled with input data on which the model is trained/plotted
        TESTING_DATA: Can test the perceptron using separate test data (if required)
        MAX_EPOCHS: Maximum number of epochs the perceptron runs for.
        PLOTLINE_COUNT: Keeps track of epoch numbers of intermediate plot separators
        PLOT_ALL: If True, it plots the plot separator for all epochs,
                  else only the final one
        bp: Input Binary Perceptron
        """
        self.X_MIN = 0
        self.X_MAX = 0
        self.TRAINING_DATA = None
        self.TESTING_DATA = None
        self.PLOTLINE_COUNT = 1
        self.MAX_EPOCHS = n_epochs
        self.PLOT_ALL = plot_all
        self.bp = bp
    
    def read_data(self):
        """
        Read training data from the given dataset
        Also reads testing data if necessary
        ---
        Contains a placeholder train dataset
        """
        self.TRAINING_DATA = [
            [-2, 7, +1],
            [1, 10, +1],
            [3, 5, +1],
            [3, 2, -1],
            [5, -2, -1]]
    
    def plot_2d_points(self, points_to_plot):
        """
        points_to_plot: list of triples of the form [xi, yi, ci]
        where ci is either -1 or +1.
        """
        xpoints = [pt[0] for pt in points_to_plot]
        self.X_MIN = min(xpoints)
        self.X_MAX = max(xpoints)
        plt.figure(figsize=(10, 6))
        ypoints = [pt[1] for pt in points_to_plot]
        classes = ['o:r' if pt[2] == -1 else 'P:b' for pt in points_to_plot]
        for (x, y, c) in zip(xpoints, ypoints, classes):
            plt.plot(x, y, c, linestyle='')
    
    def plot_separator(self, w0, w1, w2):
        """
        Add to the plot so far a line that best represents
        the current set of weights, where we are interpreting
        them as w0*x + w1*y + w2 = 0.
        x
        """
        y1 = (-w2 - w0 * self.X_MIN) / w1
        y2 = (-w2 - w0 * self.X_MAX) / w1
        if self.PLOT_ALL:
            plt.plot([self.X_MIN, self.X_MAX], [y1, y2], label='Epoch {i}'.format(i=self.PLOTLINE_COUNT))
        else:
            plt.plot([self.X_MIN, self.X_MAX], [y1, y2], label='Decision Boundary')
        self.PLOTLINE_COUNT += 1
    
    def train(self, verbose=False):
        """
        Trains the Binary perceptron
        verbose: If True, prints out the weights and changed count
                at every epoch
        """
        self.read_data()
        self.plot_2d_points(self.TRAINING_DATA)
        
        for i in range(self.MAX_EPOCHS):
            changed_count = self.bp.train_for_an_epoch(self.TRAINING_DATA)
            if changed_count == 0:
                print("Converged in ", i, " epochs.")
                print("TRAINING IS DONE")
                if not self.PLOT_ALL:
                    self.plot_separator(*self.bp.weights)
                return
            if verbose:
                print(f"changed_count= {changed_count}")
                print(f"Weights:\n{self.bp.weights}")
            if self.PLOT_ALL:
                self.plot_separator(*self.bp.weights)
        self.plot_separator(*self.bp.weights)
        print(f"Training did not converge in {self.MAX_EPOCHS} epochs.")
        
    def test(self):
        """
        If we have testing data, the child class will implement this method
        """
        pass
    
    def plot(self):
        """
        Plots the dataset as well as the binary classifier
        """
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    binary_perceptron = BinaryPerceptron(alpha=0.5)
    pbp = PlotBinaryPerceptron(binary_perceptron)
    pbp.train(verbose=True)
    pbp.plot()
