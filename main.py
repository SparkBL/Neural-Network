import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, inputs, outputs, hidden_nodes, learning_rate, epochs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights1 = np.random.randn(3, hidden_nodes)
        self.weights2 = np.random.randn(hidden_nodes, 1)
        self.error_history = []
        self.epoch_list = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0.000

    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        self.hidden_act = self.sigmoid(
            np.dot(self.inputs+self.bias, self.weights1))
        self.output_act = self.sigmoid(
            np.dot(self.hidden_act, self.weights2))

    def backpropagation(self):
        self.error = self.outputs - self.output_act

        cost2 = self.error * self.sigmoid(self.output_act, deriv=True)
        actCost = np.dot(cost2, self.weights2.T)
        cost1 = actCost * self.sigmoid(np.dot(inputs+self.bias, self.weights1))
        self.weights2 += np.dot(self.hidden_act.T, cost2)*self.learning_rate
        self.weights1 += np.dot(self.inputs.T+self.bias,
                                cost1)*self.learning_rate

    def train(self):
        for epoch in range(self.epochs):
            self.feed_forward()
            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        return self.sigmoid(
            np.dot(self.sigmoid(
                np.dot(new_input+self.bias, self.weights1)), self.weights2))


NN = NeuralNetwork(inputs, outputs, 100, 0.5, 30000)
NN.train()

for i in range(len(inputs)):
    print(NN.predict(inputs[i]), " -- Correct answer: ", outputs[i])


plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
