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
                   [0, 0, 0]
                   ])
# output data
test_inputs = np.array([[1, 0, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0]])

outputs = np.array([[0.1],
                    [0.1],
                    [0.2],
                    [0.3],
                    [0.2],
                    [0.3],
                    [0.3],
                    [0.1],
                    [0.2]
                    ])

test_outputs = np.array([[0.1],
                         [0.1],
                         [0.2],
                         [0.3]])

# create NeuralNetwork class


class NeuralNetwork:

    def __init__(self, inputs, outputs, hidden_nodes=[100, 200], learning_rate=0.5, epochs=30000):
        self.inputs = inputs
        self.outputs = outputs
        self.weights1 = np.random.randn(3, hidden_nodes[0])
        self.weights2 = np.random.randn(hidden_nodes[0], hidden_nodes[1])
        self.weights3 = np.random.randn(hidden_nodes[1], 1)
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
        self.hidden_act1 = self.sigmoid(
            np.dot(self.inputs+self.bias, self.weights1))
        self.hidden_act2 = self.sigmoid(
            np.dot(self.hidden_act1, self.weights2))
        self.output_act = self.sigmoid(
            np.dot(self.hidden_act2, self.weights3))

    def backpropagation(self):
        self.error = self.outputs - self.output_act
        weight_delta3 = self.error * self.sigmoid(self.output_act, deriv=True)
        self.weights3 += np.dot(self.hidden_act2.T, weight_delta3
                                ) * self.learning_rate
        error2 = weight_delta3 * self.weights3.T
        weight_delta2 = error2 * self.sigmoid(self.hidden_act2, deriv=True)
        self.weights2 += np.dot(self.hidden_act1.T,
                                weight_delta2) * self.learning_rate
        error1 = np.dot(weight_delta2, self.weights2.T)
        weight_delta1 = error1 * self.sigmoid(self.hidden_act1, deriv=True)
        self.weights1 += np.dot(self.inputs.T, weight_delta1
                                ) * self.learning_rate

    def train(self):
        for epoch in range(self.epochs):
            self.feed_forward()
            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input):
        return self.sigmoid(
            np.dot(self.sigmoid(
                np.dot(self.sigmoid(
                    np.dot(new_input+self.bias, self.weights1)), self.weights2)), self.weights3))


for i in range(3):
    NN = NeuralNetwork(inputs, outputs, hidden_nodes=[
        101, 7], learning_rate=0.1, epochs=30000)
    NN.train()

    for i in range(len(test_inputs)):
        print(NN.predict(test_inputs[i]),
              " -- Correct answer: ", test_outputs[i])


plt.figure(figsize=(15, 5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
