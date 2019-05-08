import tensorflow as tf
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


@dataclass(frozen=True)
class Input:
    x: object
    y: object

def regression_data(TRUE_W, TRUE_b: float, NUM_EXAMPLES: int):
    x = tf.random_normal(shape=[NUM_EXAMPLES, len(TRUE_W)])
    noise = tf.random_normal(shape=[NUM_EXAMPLES, 1])
    y = tf.matmul(x,tf.constant(TRUE_W)) + TRUE_b + noise
    return Input(x, y)

def classification_data(**args):
    data=datasets.make_classification(**args)
    return Input(
        x=tf.convert_to_tensor(data[0]),
        y=tf.reshape(tf.convert_to_tensor(data[1]), [-1, 1])
    )

def printInput(prediction, input: Input):
    plt.scatter(input.x, input.y, c='b')
    plt.scatter(input.x, prediction, c='r')
    plt.show()

def printClassificationInput(data: Input):
    # Positive Data Points
    x_pos = np.array([data.x[i] for i in range(len(data.x))
                      if data.y[i] == 1])

    # Negative Data Points
    x_neg = np.array([data.x[i] for i in range(len(data.x))
                      if data.y[i] == 0])

    # Plotting the Positive Data Points
    plt.scatter(x_pos[:, 0], x_pos[:, 1], color='blue', label='Positive')

    # Plotting the Negative Data Points
    plt.scatter(x_neg[:, 0], x_neg[:, 1], color='red', label='Negative')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Plot of given data')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    TRUE_W = [[10.0], [2]]
    TRUE_b = 5.0
    NUM_EXAMPLES = 100000000
    dataSet = regression_data(TRUE_W, TRUE_b, NUM_EXAMPLES)
    sess = tf.Session()
    print(sess.run([dataSet.output]))


