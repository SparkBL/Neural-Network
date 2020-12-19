import numpy as np  # helps with the math
import matplotlib.pyplot as plt  # to plot error during training

# input data
inputs = np.array([[1, 1, 0],
                   [1, 0, 1],
                   [0, 1, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 0, 1],
                   [0, 1, 1],
                   [0, 1, 0]])
# output data
outputs = np.array([[0.1],
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.2],
                    [0.3],
                    [0.3],
                    [0.1],
                    [0.2],
                    [0.1],
                    [0.1],
                    [0.2],
                    [0.3]])

# create NeuralNetwork class


class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs, hidden_nodes, learning_rate, epochs):
        self.inputs = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        # np.array([[.0, .0], [.0, .0], [.0, .0]])
        self.weights1 = np.random.randn(3, hidden_nodes)
        self.weights2 = np.random.randn(
            hidden_nodes, 1)  # np.array([[.0], [.0]])
        self.error_history = []
        self.epoch_list = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0.000
    # activation function ==> S(x) = 1/1+e^(-x)

    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden_act = self.sigmoid(
            np.dot(self.inputs+self.bias, self.weights1))
        self.output_act = self.sigmoid(
            np.dot(self.hidden_act, self.weights2))

    # going backwards through the network to update weights
    def backpropagation(self):
        # self.outputs - self.hidden
        self.error = self.outputs - self.output_act
        cost2 = self.error * self.sigmoid(self.output_act, deriv=True)
        actCost = np.dot(cost2, self.weights2.T)
        self.weights2 += np.dot(self.hidden_act.T, cost2)*self.learning_rate
        cost1 = actCost * self.sigmoid(np.dot(inputs+self.bias, self.weights1))
        self.weights1 += np.dot(self.inputs.T+self.bias,
                                cost1)*self.learning_rate

    # train the neural net for 25,000 iterations

    def train(self):
        for epoch in range(self.epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        return self.sigmoid(
            np.dot(self.sigmoid(
                np.dot(new_input+self.bias, self.weights1)), self.weights2))


# create neural network
NN = NeuralNetwork(inputs, outputs, 100, 0.5, 30000)
# train neural network
NN.train()

for i in range(len(inputs)):
    print(NN.predict(inputs[i]), " -- Correct answer: ", outputs[i])

# plot the error over the entire training duration
plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
