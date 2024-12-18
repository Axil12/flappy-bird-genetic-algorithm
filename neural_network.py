from math import exp, log, tanh
import random
import types

import numpy as np


class NeuralNework:
    """
    Dense neural network

    Because that neural network is going to learn through natural selection,
    there is no backpropagation. As a result, we can try some funky
    activation functions.
    """

    relu = lambda x: np.where(x > 0, x, 0)
    leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)
    step = lambda x: np.where(x > 0, 1, 0)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    softplus = lambda x: np.log(1 + np.exp(-x))
    silu = lambda x: x / (1 + np.exp(-x))  # silu(x) = x * sigmoid(x)
    gelu = (
        lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * np.pow(x, 3))))
    )
    elu = lambda x: np.where(x > 0, x, np.exp(x) - 1)
    square = lambda x: np.pow(x, 2)
    cube = lambda x: np.pow(x, 3)

    def __init__(
        self,
        nb_inputs: int,
        hidden_layers: tuple[int],
        nb_outputs: int,
        activation: types.FunctionType | None = None,
    ):
        self.nb_inputs = nb_inputs
        self.hidden_layers = hidden_layers
        self.nb_outputs = nb_outputs
        self.activation = activation or NeuralNework.relu

        self.weight_min_val = -1
        self.weight_max_val = 1

        # Weights of the input->hidden[0] layer
        self.whi = np.random.uniform(
            self.weight_min_val,
            self.weight_max_val,
            (self.hidden_layers[0], self.nb_inputs + 1),
        )

        # Weights of the input->hidden[0] layer
        self.whh_list = [
            np.random.uniform(
                self.weight_min_val,
                self.weight_max_val,
                (self.hidden_layers[i + 1], self.hidden_layers[i] + 1),
            )
            for i in range(len(self.hidden_layers) - 1)
        ]

        # Weights of the hidden[-1]->output layer
        self.woh = np.random.uniform(
            self.weight_min_val,
            self.weight_max_val,
            (self.nb_outputs, self.hidden_layers[-1] + 1),
        )

    def output(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an input vector, feeds it to the neural network, and returns an output vector.
        """
        x = np.concatenate([x, [[1]]], axis=0)  # Adding bias

        for weights in [self.whi] + self.whh_list:
            x = np.dot(weights, x)
            x = self.activation(x)
            x = np.concatenate([x, [[1]]], axis=0)  # Adding bias

        x = np.dot(self.woh, x)
        x = self.activation(x)
        return x

    def get_layers_outputs(self, x: np.ndarray) -> list[np.ndarray]:
        result = []

        x = np.concatenate([x, [[1]]], axis=0)  # Adding bias
        result.append(x)

        for weights in [self.whi] + self.whh_list:
            x = np.dot(weights, x)
            x = self.activation(x)
            x = np.concatenate([x, [[1]]], axis=0)  # Adding bias
            result.append(x)

        x = np.dot(self.woh, x)
        x = self.activation(x)
        result.append(x)
        return result

    @staticmethod
    def crossover(net_1, net_2):
        child = NeuralNework(
            net_1.nb_inputs, net_1.hidden_layers, net_1.nb_outputs, net_1.activation
        )

        for i in range(child.whi.shape[0]):
            for j in range(child.whi.shape[1]):
                if random.random() < 0.5:
                    child.whi[i][j] = net_1.whi[i][j]
                else:
                    child.whi[i][j] = net_2.whi[i][j]

        for k in range(len(child.whh_list)):
            for i in range(child.whh_list[k].shape[0]):
                for j in range(child.whh_list[k].shape[1]):
                    if random.random() < 0.5:
                        child.whh_list[k][i][j] = net_1.whh_list[k][i][j]
                    else:
                        child.whh_list[k][i][j] = net_2.whh_list[k][i][j]

        for i in range(child.woh.shape[0]):
            for j in range(child.woh.shape[1]):
                if random.random() < 0.5:
                    child.woh[i][j] = net_1.woh[i][j]
                else:
                    child.woh[i][j] = net_2.woh[i][j]

        return child

    def clone(self):
        clone = NeuralNework(
            self.nb_inputs, self.hidden_layers, self.nb_outputs, self.activation
        )
        clone.whi = np.copy(self.whi)
        for i in range(len(self.whh_list)):
            clone.whh_list[i] = np.copy(self.whh_list[i])
        clone.woh = np.copy(self.woh)
        return clone

    def mutate(self, mutation_rate: float) -> None:
        """
        Randomly mutates the weights neural network.
        The mutation_rate dictates the probability for each weight to be randomized.
        """
        for weight_matrix in [self.whi] + self.whh_list + [self.woh]:
            for i in range(0, weight_matrix.shape[0], 1):
                for j in range(0, weight_matrix.shape[1], 1):
                    if random.random() < mutation_rate:
                        weight_matrix[i][j] = random.uniform(
                            self.weight_min_val, self.weight_max_val
                        )


def main():
    net = NeuralNework(3, (5, 7, 6), 2, activation=NeuralNework.softplus)
    x = np.random.rand(3, 1)
    print(net.output(x))
    print(np.argmax(net.output(x)))


if __name__ == "__main__":
    main()
