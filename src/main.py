import numpy as np
from numpy import exp
from numpy.random import normal
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json


# Train & test data sets
x_train_g = np.empty((280, 4))
y_train_g = np.empty((280, ))
x_test_g = np.empty((40, 4))
y_test_g = np.empty((40, ))


class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype
        self.gradient_method = gradient_method

        self.init_params()

    def init_params(self):
        self.theta = []
        self.theta.append({'W0': normal(0, 0.05, (self.num_hidden, self.num_input)),
                           'b0': normal(0, 0.05, self.num_hidden)})  # hidden layer
        self.theta.append({'W1': normal(0, 0.05, (self.num_output, self.num_hidden)),
                           'b1': normal(0, 0.05, self.num_output)})  # output layer

    @staticmethod
    def softmax(x: np.ndarray):
        """ Non-linear output activation function, i.e. softmax for multi-class classification, which outputs the probabilities for each class w.r.t. the sample.

        param x: pre-activation input for output-layer
        return: softmax output (probabilities of all classes which sum to 1)
        """

        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def softplus(x: np.ndarray):
        """ Softplus activation function used for hidden layer.

        param x: preactivation value of neurons in hidden layer
        return: calculated softplus value (activated value of hidden layer)
        """
        return np.log(1 + exp(x))

    def train(self, lr, epochs):
        for epoch in range(epochs):
            for train_sample in x_train_g:
                y_hat = self.forward(train_sample)
                self.backward()

    def predict(self, x: np.ndarray):
        return self.forward(x, store_predictions=False)

    def forward(self, x: np.ndarray, store_predictions=True):
        self.theta[0]['z1'] = self.theta[0]['W0'] @ x + self.theta[0]['b0']
        self.theta[0]['a1'] = self.softplus(self.theta[0]['z1'])
        self.theta[1]['z2'] = self.theta[1]['W1'] @ self.theta[0]['a1'] + self.theta[1]['b1']
        self.theta[1]['a2'] = self.softmax(self.theta[1]['z2'])

        return self.theta[1]['a2']

    def backward(self):
        """ Perform backwards pass (backpropagate errors using partial derivatives via chain-rule)
        """



        print('x')

    def backward_old(self, y, y_hat):
        """

        :param y: target label for supervised learning
        :param y_hat: predicted value
        :return:
        """

        del_L_a2 = - np.divide(y, y_hat)  # derivative of loss w.r.t. output activation function
        del_a2_z2 = self.softmax(self.z2) - self.softmax(self.z2)**2
        del_z2_W1 = self.a1
        del_z2_a1 = self.theta['W1']
        del_a1_z1 = np.exp(self.z1) / (1 + np.exp(self.z1))
        del_z1_W0 = self.x

        # del_W1 = np.matmul((del_L_a2 * del_a2_z2).reshape(self.num_output, 1), del_z2_W1.T.reshape(1, self.num_hidden))
        # del_b1 = del_L_a2 * del_a2_z2
        # del_W0 = del_L_a2 * del_a2_z2 * del_z2_a1 * del_a1_z1 * del_z1_W0
        # del_b0 = del_L_a2 * del_a2_z2 * del_z2_a1 * del_a1_z1

        print('x')

    def export_model(self):
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            for layer in self.theta:
                json.dump({key: value.tolist() for key, value in layer.items()}, fp)


def task1():
    """ Neural Network

        Requirements for the plots:
            - ax[0] Plot showing the training loss for both variants
            - ax[1] Plot showing the training and test accuracy for both variants
    """

    # Define network parameters
    num_input = 4
    num_hidden = 16
    num_output = 3

    net_GD = NN(num_input, num_hidden, num_output, gradient_method='GD')  # create NN with the steepest gradient descent
    # y_hat = net_GD.forward(x_test_g[0, :])
    # net_GD.backward(y_train_g[0], y_hat)
    net_GD.train(lr=0.01, epochs=350)  # lr = {0.1, 0.001}

    # Model using Nesterovs method
    # net_NAG = NN(num_input, num_hidden, num_output, gradient_method='NAG')

    # Export models
    net_GD.export_model()
    # net_NAG.export_model()

    return __create_plots()


def __create_plots():
    # Configure plot
    fig = plt.figure(figsize=[12, 6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))
    axs[0].set_title('Loss')
    axs[0].grid()
    axs[1].set_title('Accuracy')
    axs[1].grid()
    return fig


def __load_data_set():
    global x_train_g, y_train_g, x_test_g, y_test_g

    # Load train data
    with np.load('data_train.npz') as data_set:
        x_train_g = data_set['x']
        y_train_g = data_set['y']

    # Load test data
    with np.load('data_test.npz') as data_set:
        x_test_g = data_set['x']
        y_test_g = data_set['y']

    print(f'Training set with {x_train_g.shape[0]} data samples.')
    print(f'Test set with {x_test_g.shape[0]} data samples.')


if __name__ == '__main__':
    __load_data_set()
    tasks = [task1]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        retval = task()
        fig = retval[0] if type(retval) is tuple else retval
        pdf.savefig(fig)
    pdf.close()
