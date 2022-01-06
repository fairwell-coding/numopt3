import numpy as np
from numpy import exp
from numpy.random import normal
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json

# Train & test data sets
x_train_g = np.empty((280, 4))
y_train_g = np.empty((280,))
x_test_g = np.empty((40, 4))
y_test_g = np.empty((40,))


class NN(object):
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, momentum=0.0, dtype=np.float32):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype
        self.gradient_method = gradient_method
        self.last_gradient_update = 0

        self.momentum = momentum
        self.last_update_W1 = np.empty((3, 16))
        self.last_update_b1 = np.empty((3,))
        self.last_update_W0 = np.empty((16, 4))
        self.last_update_b0 = np.empty((16,))

        self.init_params()

    def init_params(self):
        """ Create layers and their corresponding learnable model parameters (theta)

        """

        self.layers = []
        self.layers.append({'W0': normal(0, 0.05, (self.num_hidden, self.num_input)),
                            'b0': normal(0, 0.05, self.num_hidden)})  # hidden layer
        self.layers.append({'W1': normal(0, 0.05, (self.num_output, self.num_hidden)),
                            'b1': normal(0, 0.05, self.num_output)})  # output layer

    @staticmethod
    def softmax(x: np.ndarray):
        """ Non-linear output activation function, i.e. softmax for multi-class classification, which outputs the probabilities for each class w.r.t. the sample.

        param x: pre-activation input for output-layer
        return: softmax output (probabilities of all classes which sum to 1)
        """

        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def softmax_derivative(x: np.ndarray):
        """ Calculates the derivative of the softmax output function

        param x: backpropagated error (delta) from previous layer
        return: softmax derivative of x
        """

        # s = NN.softmax(x).reshape(-1, 1)  # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        # return np.diagflat(s) - np.dot(s, s.T)

        return x  # TODO: dummy replacement to continue implementation

    @staticmethod
    def softplus(x: np.ndarray):
        """ Softplus activation function used for hidden layer.

        param x: preactivation value of neurons in hidden layer
        return: calculated softplus value (activated value of hidden layer)
        """
        return np.log(1 + exp(x))

    @staticmethod
    def softplus_derivative(x: np.ndarray):
        """ Calculates the derivative of the softplus activation function for the hidden layer

        param x: backpropagated error (delta) from previous layer
        return: softplus derivative of x
        """

        return np.exp(x) / (1 + np.exp(x))  # logistic function

    def train(self, lr, epochs):
        for epoch in range(epochs):
            for sample_index in range(x_train_g.shape[0]):  # using single batch (i.e. batch_size = 1)
                mini_batch_x = x_train_g[sample_index]
                mini_batch_y = self.__encode_output(y_train_g[sample_index])  # converts the target label to the required network output format (i.e. 3 neurons of last layer)

                self.forward(mini_batch_x)

                if self.gradient_method == 'GD':
                    self.backward_GD(mini_batch_x, mini_batch_y)
                    self.steepest_descent(lr)
                elif self.gradient_method == 'NAG':
                    self.backward_NAG(mini_batch_x, mini_batch_y)
                    self.nesterov_accelerated_gradient(lr)

    def forward(self, x: np.ndarray):
        self.layers[0]['pre_activation'] = self.layers[0]['W0'] @ x + self.layers[0]['b0']
        self.layers[0]['output'] = self.softplus(self.layers[0]['pre_activation'])
        self.layers[1]['pre_activation'] = self.layers[1]['W1'] @ self.layers[0]['output'] + self.layers[1]['b1']
        self.layers[1]['output'] = self.softmax(self.layers[1]['pre_activation'])

        return self.layers[1]['output']

    def backward_GD(self, x: np.ndarray, y: np.ndarray):
        """ Perform backwards pass by computing gradients w.r.t. the model parameters (backpropagate errors through the network using partial derivatives via chain-rule)

        param x: sample used for current forward pass
        param y: supervised label
        """

        # Calculate gradients for parameters of output layer
        del_L_a2 = - y / self.layers[1]['output']  # derivative of loss function w.r.t. predicted output (i.e. activation of output layer: a2)
        del_a2_z2 = self.softmax_derivative(self.layers[1]['pre_activation'])  # derivative of activated output of output layer (a2) w.r.t. its weighted input (z2)
        delta_1 = del_L_a2 * del_a2_z2  # = delta_1, b1 | (del_L / del_a2) * (del_a2 / del_z2): derivative of loss (L) w.r.t. pre-activation of output layer (z2)
        del_L_W1 = np.outer(delta_1, self.layers[0]['output'].T)  # W1 | delta_1 * a1.T: derivative of loss (L) w.r.t. weight matrix of output layer (W1)

        # Calculate gradients for parameters of hidden layer
        del_L_a1 = np.matmul(self.layers[1]['W1'].T, delta_1)  # derivative of loss function w.r.t. activated output of hidden layer (i.e. a1)
        del_a1_z1 = self.softplus_derivative(self.layers[0]['pre_activation'])  # derivative of activated output of hidden layer (a1) w.r.t. its weighted input (z1)
        delta_0 = del_L_a1 * del_a1_z1  # = delta_2, b0 | (del_L / del_z1): derivative of loss (L) w.r.t. pre-activation of hidden layer (z1)
        del_L_W0 = np.outer(delta_0, x.T)  # W0 | delta_0 * x.T: derivative of loss (L) w.r.t. weight matrix of hidden layer (W0)

        # Store calculated gradients in their respective network layers
        self.layers[1]['W1_grad'] = del_L_W1
        self.layers[1]['b1_grad'] = delta_1
        self.layers[0]['W0_grad'] = del_L_W0
        self.layers[0]['b0_grad'] = delta_0

    def backward_NAG(self, x: np.ndarray, y: np.ndarray):
        """ Perform backwards pass by computing gradients w.r.t. the model parameters (backpropagate errors through the network using partial derivatives via chain-rule). This method calculates the look-ahead

        param x: sample used for current forward pass
        param y: supervised label
        """

        # Calculate look-ahead weights
        W1_look_ahead = self.layers[1]['W1'] - self.momentum * self.last_update_W1
        b1_look_ahead = self.layers[1]['b1'] - self.momentum * self.last_update_b1
        W0_look_ahead = self.layers[0]['W0'] - self.momentum * self.last_update_W0
        b0_look_ahead = self.layers[0]['b0'] - self.momentum * self.last_update_b0

        # Calculate gradients for parameters of output layer
        delta_1 = b1_look_ahead  # use look-head instead of calculating the current value
        del_L_W1 = np.outer(delta_1, self.layers[0]['output'].T)  # W1 | delta_1 * a1.T: derivative of loss (L) w.r.t. weight matrix of output layer (W1)

        # Calculate gradients for parameters of hidden layer
        delta_0 = b0_look_ahead
        del_L_W0 = np.outer(delta_0, x.T)  # W0 | delta_0 * x.T: derivative of loss (L) w.r.t. weight matrix of hidden layer (W0)

        # Store calculated gradients in their respective network layers
        self.layers[1]['W1_grad'] = del_L_W1
        self.layers[1]['b1_grad'] = delta_1
        self.layers[0]['W0_grad'] = del_L_W0
        self.layers[0]['b0_grad'] = delta_0

    def steepest_descent(self, lr):
        """ Optimize network weights based on steepest gradient descent algorithm to update model weights for current training iteration

        param lr: learning rate used for optimization step
        """

        self.layers[1]['W1'] -= lr * self.layers[1]['W1_grad']
        self.layers[1]['b1'] -= lr * self.layers[1]['b1_grad']
        self.layers[0]['W0'] -= lr * self.layers[0]['W0_grad']
        self.layers[0]['b0'] -= lr * self.layers[0]['b0_grad']

    def nesterov_accelerated_gradient(self, lr):
        """ Optimize network weights based on Nesterov accelerated gradient method to update model weights for current training iteration

        param lr: learning rate used for optimization step
        """

        # update gradients based on look-ahead gradients
        self.layers[1]['W1'] -= self.momentum * self.last_update_W1 + lr * self.layers[1]['W1_grad']
        self.layers[1]['b1'] -= self.momentum * self.last_update_b1 + lr * self.layers[1]['b1_grad']
        self.layers[0]['W0'] -= self.momentum * self.last_update_W0 + lr * self.layers[0]['W0_grad']
        self.layers[0]['b0'] -= self.momentum * self.last_update_b0 + lr * self.layers[0]['b0_grad']

        # store last gradient updates needed for next optimization step
        self.last_update_W1 = self.layers[1]['W1']
        self.last_update_b1 = self.layers[1]['b1']
        self.last_update_W0 = self.layers[0]['W0']
        self.last_update_b0 = self.layers[0]['b0']

    def __encode_output(self, y):
        target_label = [0 for i in range(self.num_output)]
        target_label[y] = 1
        return np.asarray(target_label, dtype=self.dtype)

    def export_model(self):
        with open(f'model_{self.gradient_method}.json', 'w') as fp:
            for layer in self.layers:
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

    # y_hat = net_GD.forward(x_test_g[0, :])
    # net_GD.backward(y_train_g[0], y_hat)
    # net_GD = NN(num_input, num_hidden, num_output, gradient_method='GD')  # create NN with the steepest gradient descent
    # net_GD.train(lr=0.01, epochs=350)  # lr = {0.1, 0.001}

    # Model using Nesterovs method
    net_NAG = NN(num_input, num_hidden, num_output, gradient_method='NAG', momentum=0.9)
    net_NAG.train(lr=0.01, epochs=350)

    # Export models
    # net_GD.export_model()
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
