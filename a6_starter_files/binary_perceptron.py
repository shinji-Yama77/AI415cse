"""binary_perceptron.py
One of the starter files for use in CSE 415, Autumn 2023
Assignment 6.
Complete this python file.

This program can be run from the given Python program
called run_2_class_2_feature_iris_data.py.
"""


def student_name():
    return "Shinji Yamashita"  # Replace with your own name here


class BinaryPerceptron:
    """
    Class representing the Binary Perceptron
    ---
    It is an algorithm that can learn a binary classifier
    """
    
    def __init__(self, weights=None, alpha=0.5):
        """
        Initialize the Binary Perceptron
        ---
        weights: Weight vector of the form [w_0, w_1, ..., w_{n-1}, bias_weight]
        alpha: Learning rate
        """
        if weights is None:
            self.weights = [0, 0, 0]
        else:
            self.weights = weights[:]
        self.alpha = alpha
    
    def classify(self, x_vector):
        """
        Method that classifies a given data point into one of 2 classes
        ---
        Inputs:
        x_vector = [x_0, x_1, ..., x_{n-1}]
        Note: y (correct class) is not part of the x_vector.

        Returns:
        y_hat: Predicted class
              +1 if the current weights classify x_vector as positive i.e. the required dot product must be >=0,
        else  -1 if it is classified as negative.
        """
        # ADD YOUR CODE HERE
        print(x_vector)
        predicted_y = -1
        dot_product = 0 
        for i, x_feature_val in enumerate(x_vector):
            dot_product += x_feature_val*self.weights[i]
        dot_product+= self.weights[-1]
        if dot_product >= 0:
            predicted_y = 1
        return predicted_y
    def train_with_one_example(self, x_vector, y):
        """
        Method that updates the model weights using a particular training example (x_vector,y)
        and returns whether the model weights were actually changed or not
        ---
        Inputs:
        x_vector: Feature vector, same as method classify
        y: Actual class of x_vector
            +1 if x_vector represents a positive example,
        and -1 if it represents a negative example.
        Returns:
        weight_changed: True if there was a change in the weights
                        else False
        """
        # ADD YOUR CODE HERE
        # if diff = zero
        # only updates the weights if there is a problem with the classification
        # 
        weight_changed = False
        curr_weights = self.weights
        new_weights = list(range(len(curr_weights)))
        diff = y - self.classify(x_vector)
        if diff != 0:
            for i in range(len(new_weights)-1):
                new_weights[i] = curr_weights[i] + self.alpha * y * x_vector[i]
            new_weights[-1] = curr_weights[-1] + self.alpha * y
            self.weights = new_weights
            for i in range(len(new_weights)):
                if new_weights[i] != curr_weights[i]:
                    weight_changed = True
        return weight_changed


    def train_for_an_epoch(self, training_data):
        """
        Method that goes through the given training examples once, in the order supplied,
        passing each one to train_with_one_example.
        ---
        Input:
        training_data: Input training data
        [[x_vector_1, y_1], [x_vector_2, y_2], ...]
        where each x_vector is concatenated with the corresponding y value.

        Returns:
        changed_count: Return the number of weight updates.
        (If zero, then training has converged.)
        """
        # ADD YOUR CODE HERE
        changed_count = 0
        for feature_vector in training_data:
            changed_count += int(self.train_with_one_example(feature_vector[:-1], feature_vector[-1]))
        return changed_count


def sample_test():
    """
    May be useful while developing code
    Trains the binary perceptron using a synthetic training set
    Prints the weights obtained after training
    """
    DATA = [
        [-2, 7, +1],
        [1, 10, +1],
        [3, 2, -1],
        [5, -2, -1]]
    bp = BinaryPerceptron()
    print("Training Binary Perceptron for 3 epochs.")
    for i in range(3):
        bp.train_for_an_epoch(DATA)
    print("Binary Perceptron weights:")
    print(bp.weights)
    print("Done.")


if __name__ == '__main__':
    sample_test()
