import random
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.1
STEPS = 10000
MIN_X1 = -10
MAX_X1 = 10
MIN_X2 = -35
MAX_X2 = 35


def split_fn(x):
    return 3 * x + 2


def train_generate_point():
    data = []
    for i in range(1000):
        x1 = random.uniform(MIN_X1, MAX_X1)
        x2 = random.uniform(MIN_X2, MAX_X2)
        # y = 4x - 5
        delta = x2 - split_fn(x1)
        delta = np.sign(delta)
        data.append([x1, x2, delta])
    return data


def test_generate_point():
    data = []
    for i in range(1000):
        x1 = random.uniform(MIN_X1, MAX_X1)
        x2 = random.uniform(MIN_X2, MAX_X2)
        data.append([x1, x2])
    return data


def show_scatter(data):
    x1 = [item[0] for item in data if item[2] == 1]
    x2 = [item[1] for item in data if item[2] == 1]
    plt.scatter(x1, x2, color="red")
    x1 = [item[0] for item in data if item[2] == -1]
    x2 = [item[1] for item in data if item[2] == -1]
    plt.scatter(x1, x2, color="green")
    x1 = [item[0] for item in data if item[2] == 0]
    x2 = [item[1] for item in data if item[2] == 0]
    plt.scatter(x1, x2, color="blue")
    line = np.linspace(-10, 10)
    plt.plot(line, split_fn(line), color="black")
    plt.show()


def perceptron(train_data, w):
    for i in train_data:
        suma_weight = w[0] * i[0] + w[1] * i[1]
        suma_weight = np.sign(suma_weight)

        error = i[2] - suma_weight
        for x in range(len(w)):
            w[x] = w[x] + error * i[x] * LEARNING_RATE
    return w


def testing(w, test_data):
    new_data = []
    for i in test_data:
        suma_weight = np.dot(w, i)
        suma_weight = np.sign(suma_weight)
        new_data.append([i[0], i[1], suma_weight])
    return new_data


def main():
    train_data = train_generate_point()
    test_data = test_generate_point()
    w = [random.random() for x in range(2)]
    for _ in range(STEPS):
        w = perceptron(train_data, w)
    test_data = testing(w, test_data)
    show_scatter(test_data)


if __name__ == "__main__":
    main()
