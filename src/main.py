import numpy as np
from numpy import exp
from numpy.random import normal
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json

from scipy.special import softmax


x_train_g = np.empty((40, 4))
y_train_g = np.empty((280, 4))
x_test_g = np.empty((40,))
y_test_g = np.empty((280,))


class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype
        self.gradient_method = gradient_method

        self.init_params()

    def init_params(self):
        self.theta = {'W0': normal(0, 0.05, (self.num_hidden, self.num_input)),
                      'W1': normal(0, 0.05, (self.num_output, self.num_hidden)),
                      'b0': normal(0, 0.05, self.num_hidden),
                      'b1': normal(0, 0.05, self.num_output)}

    @staticmethod
    def activation_h(z: np.ndarray):
        return np.log(1 + exp(z))

    def forward(self, x: np.ndarray):
        return softmax(self.theta['W1'] @ self.activation_h(self.theta['W0'] @ x + self.theta['b0']) + self.theta['b1'])

    def export_model(self):
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            json.dump({key: value.tolist() for key, value in self.theta.items()}, fp)


def task1():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss for both variants
            - ax[1] Plot showing the training and test accuracy for both variants
    """

    num_input = 4
    num_hidden = 16
    num_output = 3

    # Create the models
    # Model using steepest descent
    net_GD = NN(num_input, num_hidden, num_output, gradient_method='GD')
    y_hat = net_GD.forward(x_test_g[0, :])

    # Model using Nesterovs method
    net_NAG = NN(num_input, num_hidden, num_output, gradient_method='NAG')

    net_GD.export_model()
    net_NAG.export_model()

    # Configure plot
    fig = plt.figure(figsize=[12,6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))

    axs[0].set_title('Loss')
    axs[0].grid()

    axs[1].set_title('Accuracy')
    axs[1].grid()
    return fig


if __name__ == '__main__':

    # load the data set
    with np.load('data_train.npz') as data_set:
        # get the training data
        x_train_g = data_set['x']
        y_train_g = data_set['y']

    with np.load('data_test.npz') as data_set:
        # get the test data
        x_test_g = data_set['x']
        y_test_g = data_set['y']

    print(f'Training set with {x_train_g.shape[0]} data samples.')
    print(f'Test set with {x_test_g.shape[0]} data samples.')

    tasks = [task1]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()

    
