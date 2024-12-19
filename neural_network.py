from math import exp, log, tanh
import random
import types
from typing import Self
import os

import numpy as np


def isalambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


class NeuralNetwork:
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
        dimensions: tuple[int],
        activation: types.FunctionType | None = None,
    ):
        self.weight_min_val = -1
        self.weight_max_val = 1

        self.activation = activation or NeuralNetwork.relu
        self.dims = dimensions
        self.weights = [
            np.random.uniform(
                self.weight_min_val,
                self.weight_max_val,
                (self.dims[i + 1], self.dims[i] + 1),
            )
            for i in range(len(self.dims) - 1)
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Takes an input vector, feeds it to the neural network, and returns an output vector.
        """
        for weights in self.weights:
            x = np.concatenate([x, [[1]]], axis=0)  # Adding bias
            x = np.dot(weights, x)
            x = self.activation(x)
        return x

    def get_layers_outputs(self, x: np.ndarray) -> list[np.ndarray]:
        result = []
        for weights in self.weights:
            x = np.concatenate([x, [[1]]], axis=0)  # Adding bias
            result.append(x)
            x = np.dot(weights, x)
            x = self.activation(x)
        result.append(x)
        return result

    @staticmethod
    def crossover(net_1, net_2):
        child = NeuralNetwork(net_1.dims, net_1.activation)

        for k in range(len(child.weights)):
            for i in range(child.weights[k].shape[0]):
                for j in range(child.weights[k].shape[1]):
                    if random.random() < 0.5:
                        child.weights[k][i][j] = net_1.weights[k][i][j]
                    else:
                        child.weights[k][i][j] = net_2.weights[k][i][j]

        return child

    def clone(self):
        clone = NeuralNetwork(self.dims, self.activation)

        for i in range(len(self.weights)):
            clone.weights[i] = np.copy(self.weights[i])

        return clone

    def mutate(self, mutation_rate: float) -> None:
        """
        Randomly mutates the weights neural network.
        The mutation_rate dictates the probability for each weight to be randomized.
        """
        for weight_matrix in self.weights:
            for i in range(0, weight_matrix.shape[0], 1):
                for j in range(0, weight_matrix.shape[1], 1):
                    if random.random() < mutation_rate:
                        weight_matrix[i][j] = random.uniform(
                            self.weight_min_val, self.weight_max_val
                        )


    def save(self, directory: str, filename: str | None = None) -> None:
        # The following lines aim to convert self.activation into a string representing its name
        activation_function_name = None
        possible_activations = [
            (key, val) for key, val in NeuralNetwork.__dict__.items() if isalambda(val)
        ]
        for act_name, act_func in possible_activations:
            if self.activation is act_func:
                activation_function_name = act_name
                break
        if activation_function_name is None:
            raise ValueError("Couldn't identify the activation function")
        

        if filename is None:
            dims_str = "-".join([str(dim) for dim in self.dims])
            filename = f"{dims_str}_{activation_function_name}.npz"

        file_path = os.path.join(directory, filename)
        np.savez(
            file_path,
            np.array(activation_function_name),
            *self.weights
            )

        return file_path

    @staticmethod
    def load(filename) -> Self:
        matrices = []
        with open(filename, "rb") as f:
            npz_file = np.load(f)
            matrices = [npz_file[key] for key in npz_file.keys()]

        func_name = str(matrices[0])
        weights = matrices[1:]

        output_dim = weights[-1].shape[0]
        dims = [w.shape[1] - 1 for w in weights] + [output_dim]

        possible_activations = {
            key: val for key, val in NeuralNetwork.__dict__.items() if isalambda(val)
        }
        try:
            act_func = possible_activations[func_name]
        except KeyError:
            raise KeyError(
                "Couldn't infer the activation function of the loaded neural network."
            )

        net = NeuralNetwork(dims, activation=act_func)
        net.weights = weights

        return net


def main():
    net = NeuralNetwork((6, 5, 7, 6, 12, 10, 2, 4), activation=NeuralNetwork.leaky_relu)
    x = np.random.rand(6, 1)
    net.clone()
    NeuralNetwork.crossover(net, net)
    net.mutate(0.5)
    print(net(x))
    #net.save("saved_neural_networks", "test_save")
    #print(NeuralNetwork.load("saved_neural_networks\\test_save.npz"))


if __name__ == "__main__":
    main()
