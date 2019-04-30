import tensorflow as tf
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Input:
    inputs: object
    noise: object
    outputs: object

class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))

def inputData(TRUE_W: int, TRUE_b: int, NUM_EXAMPLES: int):
    inputs = tf.random_normal(shape=[NUM_EXAMPLES])
    noise = tf.random_normal(shape=[NUM_EXAMPLES])
    outputs = inputs * TRUE_W + TRUE_b + noise
    return Input(inputs,noise, outputs)

def printInput(prediction, input: Input):
    plt.scatter(input.inputs, input.outputs, c='b')
    plt.scatter(input.inputs, prediction, c='r')
    plt.show()

def training(epochs: int, model: Model, learning_rate, trainData):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gs = tf.Variable(0)
    i=tf.Variable(0)

    def condition(i): i<epochs

    def body(i):
        current_loss = loss(model(trainData.input), trainData.output)
        train_op = optimizer.minimize(current_loss, global_step=gs)
        return tf.tuple([tf.add(i, 1)], control_inputs=[train_op])

    print("hello")

    return  tf.while_loop(condition, body, [i])

model = Model()
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data=inputData(TRUE_W, TRUE_b, NUM_EXAMPLES)
    input, noise, output,predictions=sess.run([data.inputs,data.noise, data.outputs, model(data.inputs)])
    printInput(predictions, Input(input,noise,output))
    print('Current loss: '),
    print(sess.run(loss(predictions, output)))
    print(sess.run(training(1000, model, 0.1,Input(input,noise,output))))


# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(input), output)
  tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(current_loss)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()

