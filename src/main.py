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
    def __init__(self, num_input: int, num_hidden: int, num_output: int, gradient_method: str, dtype=np.float32, batch_size=1):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.dtype = dtype

        self.gradient_method = gradient_method
        self.last_gradient_update = 0

        self.batch_size = batch_size
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []

        self.momentum = 0.0
        self.last_update_W1 = np.zeros((3, 16))
        self.last_update_b1 = np.zeros((3,))
        self.last_update_W0 = np.zeros((16, 4))
        self.last_update_b0 = np.zeros((16,))

        self.init_params()

    def init_params(self):
        """ Create layers and their corresponding learnable model parameters (theta)

        """

        self.layers = []
        self.layers.append({'W0': normal(0, 0.05, (self.num_hidden, self.num_input)),
                            'b0': normal(0, 0.05, self.num_hidden)})  # hidden layer
        self.layers.append({'W1': normal(0, 0.05, (self.num_output, self.num_hidden)),
                            'b1': normal(0, 0.05, self.num_output)})  # output layer

    def get_train_loss_for_epochs(self):
        return self.train_loss

    def get_train_acc_for_epochs(self):
        return self.train_acc

    def get_test_acc_for_epochs(self):
        return self.test_acc

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

        s = NN.softmax(x).reshape(-1, 1)  # reshape vector to matrix
        return np.diagflat(s) - np.dot(s, s.T)  # for i = j we have the same diagonal y_hat - y_hat**2 | for i != j we have - y_hat**2

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
            self.train_model(lr)
            self.calculate_test_accuracy()

    def calculate_test_accuracy(self):
        """ Calculates test accuracy for whole test set for current epoch (i.e. how the model performs after the current training epoch on the test data).

        """

        test_acc_for_epoch = 0

        for sample_index in range(x_test_g.shape[0]):
            x = x_test_g[sample_index]
            y = y_test_g[sample_index]
            y_encoded = self.__encode_output(y)

            y_hat = self.forward(x)

            # Calculate test accuracy
            if y == np.argmax(y_hat):
                test_acc_for_epoch += 1

        self.test_acc.append(test_acc_for_epoch / y_test_g.shape[0])

    def train_model(self, lr):
        train_acc_for_epoch = 0
        y_hat_epoch = np.zeros((y_train_g.shape[0], self.num_output))  # model predictions of current epoch
        y_encoded_epoch = np.zeros((y_train_g.shape[0], self.num_output))  # supervised labels of current epoch

        for sample_index in range(x_train_g.shape[0]):  # using single batch (i.e. batch_size = 1)
            x = x_train_g[sample_index]
            y = y_train_g[sample_index]
            y_encoded = self.__encode_output(y)  # converts the target label to the required network output format (i.e. 3 neurons of last layer)

            y_hat = self.forward(x)

            # # Calculate train accuracy
            if y == np.argmax(y_hat):
                train_acc_for_epoch += 1

            # Track model predictions for all batches over epoch
            y_hat_epoch[sample_index, :] = y_hat
            y_encoded_epoch[sample_index, :] = y_encoded

            self.backpropagation(x, y_encoded)
            if self.gradient_method == 'GD':
                self.steepest_descent(lr)
            elif self.gradient_method == 'NAG':
                self.backward_NAG(x, y_encoded)
                self.nesterov_accelerated_gradient(lr)

        self.train_acc.append(train_acc_for_epoch / x_train_g.shape[0])  # Track training accuracy over epochs
        self.train_loss.append(self.calculate_loss(y_encoded_epoch, y_hat_epoch))

    def forward(self, x: np.ndarray):
        self.layers[0]['pre_activation'] = self.layers[0]['W0'] @ x + self.layers[0]['b0']
        self.layers[0]['output'] = self.softplus(self.layers[0]['pre_activation'])
        self.layers[1]['pre_activation'] = self.layers[1]['W1'] @ self.layers[0]['output'] + self.layers[1]['b1']
        self.layers[1]['output'] = self.softmax(self.layers[1]['pre_activation'])

        return self.layers[1]['output']  # return network prediction vector

    def forward_approx_fprime(self, x: np.ndarray, index: int):  # TODO: use for verification of partial derivatives
        return self.forward(x)[index]

    def calculate_loss(self, y: np.ndarray, y_hat: np.ndarray):
        total_sample_loss = - np.sum(y * np.log(y_hat))
        average_loss = 1 / y_train_g.shape[0] * total_sample_loss

        return average_loss

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        """ Perform backwards pass by computing gradients w.r.t. the model parameters (backpropagate errors through the network using partial derivatives via chain-rule)

        param x: sample used for current forward pass
        param y: supervised label
        """

        # Calculate gradients for parameters of output layer
        del_L_a2 = - y / self.layers[1]['output']  # derivative of loss function w.r.t. predicted output (i.e. activation of output layer: a2)
        del_a2_z2 = self.softmax_derivative(self.layers[1]['pre_activation'])  # derivative of activated output of output layer (a2) w.r.t. its weighted input (z2)
        delta_1 = np.matmul(del_a2_z2, del_L_a2)  # = delta_1, b1 | (del_L / del_a2) * (del_a2 / del_z2): derivative of loss (L) w.r.t. pre-activation of output layer (z2)
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

        # Calculate adaptive momentum value
        momentum_updated = (1 + np.sqrt(1 + 4 * self.momentum**2)) / 2

        # Calculate look-ahead weights
        self.layers[1]['W1_intermediate'] = self.layers[1]['W1'] + (self.momentum - 1) / momentum_updated * (self.layers[1]['W1'] - self.last_update_W1)
        self.layers[1]['b1_intermediate'] = self.layers[1]['b1'] + (self.momentum - 1) / momentum_updated * (self.layers[1]['b1'] - self.last_update_b1)
        self.layers[0]['W0_intermediate'] = self.layers[0]['W0'] + (self.momentum - 1) / momentum_updated * (self.layers[0]['W0'] - self.last_update_W0)
        self.layers[0]['b0_intermediate'] = self.layers[0]['b0'] + (self.momentum - 1) / momentum_updated * (self.layers[0]['b0'] - self.last_update_b0)

        # Update momentum value for next iteration
        self.momentum = momentum_updated

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
        self.layers[1]['W1'] = self.layers[1]['W1_intermediate'] - lr * self.layers[1]['W1_grad']
        self.layers[1]['b1'] = self.layers[1]['b1_intermediate'] - lr * self.layers[1]['b1_grad']
        self.layers[0]['W0'] = self.layers[0]['W0_intermediate'] - lr * self.layers[0]['W0_grad']
        self.layers[0]['b0'] = self.layers[0]['b0_intermediate'] - lr * self.layers[0]['b0_grad']

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
                json.dump({key: value.tolist() for key, value in layer.items() if key in ['W0', 'b0', 'W1', 'b1']}, fp)


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
    num_epochs = 350

    net_GD = __train_with_steepest_descent(num_epochs, num_hidden, num_input, num_output)
    net_NAG = __train_with_nesterov_accelerated_gradient(num_epochs, num_hidden, num_input, num_output)

    return __create_plots(net_GD, net_NAG)


def __train_with_nesterov_accelerated_gradient(num_epochs, num_hidden, num_input, num_output):
    net_NAG = NN(num_input, num_hidden, num_output, gradient_method='NAG')
    net_NAG.train(lr=0.01, epochs=num_epochs)  # lr = {0.1, 0.001}

    train_losses = net_NAG.get_train_loss_for_epochs()
    train_accuracies = net_NAG.get_train_acc_for_epochs()
    test_accuracies = net_NAG.get_test_acc_for_epochs()

    net_NAG.export_model()

    return net_NAG


def __train_with_steepest_descent(num_epochs, num_hidden, num_input, num_output):
    net_GD = NN(num_input, num_hidden, num_output, gradient_method='GD')
    net_GD.train(lr=0.01, epochs=num_epochs)  # lr = {0.1, 0.001}

    train_losses = net_GD.get_train_loss_for_epochs()
    train_accuracies = net_GD.get_train_acc_for_epochs()
    test_accuracies = net_GD.get_test_acc_for_epochs()

    net_GD.export_model()

    return net_GD


def __create_plots(net_GD: NN, net_NAG: NN):
    fig = plt.figure(figsize=[12, 6])
    axs = []
    axs.append(fig.add_subplot(121))
    axs.append(fig.add_subplot(122))
    axs[0].set_title('Loss')
    axs[0].grid()

    epochs = np.arange(1, 350 + 1)  # 350 epochs

    GD_train_losses = net_GD.get_train_loss_for_epochs()
    NAG_train_losses = net_NAG.get_train_loss_for_epochs()

    sg_loss, = axs[0].semilogy(epochs, GD_train_losses, color="green", label="Steepest Descent")
    nag_loss, = axs[0].semilogy(epochs, NAG_train_losses, color="blue", label="Nesterov Accelerated Gradient")
    axs[0].legend(handles=[sg_loss, nag_loss])

    GD_train_acc = net_GD.get_train_acc_for_epochs()
    NAG_train_acc = net_NAG.get_train_acc_for_epochs()

    sg_acc, = axs[1].plot(epochs, GD_train_acc, color="green", label="Steepest Descent")
    nag_acc, = axs[1].plot(epochs, NAG_train_acc, color="blue", label="Nesterov Accelerated Gradient")
    axs[1].legend(handles=[sg_acc, nag_acc])

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
