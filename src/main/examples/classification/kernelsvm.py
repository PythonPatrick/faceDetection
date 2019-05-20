from src.main.model.model import Config, Parameters
from src.main.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from src.main.model.classification.kernelsvm import SVM


# model configurations
config = Config(feature_num=2, batch_size=100, learning_rate=0.01, epoche=1000)

#data
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=.5,noise=.1)
y_vals = np.array([1 if y==1 else -1 for y in y_vals])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

dataset=Dataset(tf.convert_to_tensor(x_vals, dtype=tf.float32), tf.reshape(tf.convert_to_tensor(y_vals, dtype=tf.float32), [-1,1]), batch_size=100)

# model
sess = tf.Session()
model = SVM(dataset=dataset, config=config, parameters=Parameters())
sess.run(tf.global_variables_initializer())
model.training(session=sess)
print(sess.run([model.weights, model.bias]))

x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
X=tf.convert_to_tensor(x_vals, dtype=tf.float32)
Y=tf.convert_to_tensor(y_vals, dtype=tf.float32)
kernel=model.kernel(X=X, Y=tf.convert_to_tensor(grid_points, dtype=tf.float32), gamma=-50.)
grid_predictions = sess.run(model.predictions(Y, kernel))
grid_predictions = grid_predictions.reshape(xx.shape)

plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired,
alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()


