import math
import random
from config import NN_LAYERS

def _relu(x: float) -> float:
    return x if x > 0 else 0.0

def _sigmoid(x: float) -> float:
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def _apply(func, vec):
    return [func(v) for v in vec]

def _mat_vec(mat, vec):
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]

def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

class NeuralNetwork:
    def __init__(self, layer_sizes=None):
        self.layer_sizes = layer_sizes or NN_LAYERS
        self.weights: list[list[list[float]]] = []
        self.biases:  list[list[float]]       = []
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit   = math.sqrt(6.0 / (fan_in + fan_out))
            w = [[random.uniform(-limit, limit) for _ in range(fan_in)]
                 for _ in range(fan_out)]
            b = [0.0] * fan_out
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs: list[float]) -> list[float]:
        x = list(inputs)
        num_hidden_layers = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = _vec_add(_mat_vec(w, x), b)
            if i < num_hidden_layers:
                x = _apply(_relu, x)
            else:
                x = _apply(_sigmoid, x)
        return x

    def get_flat(self) -> list[float]:
        flat = []
        for w, b in zip(self.weights, self.biases):
            for row in w:
                flat.extend(row)
            flat.extend(b)
        return flat

    def set_flat(self, flat: list[float]):
        idx = 0
        for w, b in zip(self.weights, self.biases):
            for row in w:
                for j in range(len(row)):
                    row[j] = flat[idx];  idx += 1
            for i in range(len(b)):
                b[i] = flat[idx];  idx += 1

    def clone(self) -> "NeuralNetwork":
        nn = NeuralNetwork(self.layer_sizes)
        nn.set_flat(self.get_flat())
        return nn
