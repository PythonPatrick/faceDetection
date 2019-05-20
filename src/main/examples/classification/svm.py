from src.main.model.model import Config, Parameters
from src.main.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from src.main.model.classification.svm import SVM


# model configurations
config = Config(feature_num=2, batch_size=50, learning_rate=0.01, epoche=1000)


#data
iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
dataset=Dataset(tf.convert_to_tensor(x_vals_train, dtype=tf.float32), tf.reshape(tf.convert_to_tensor(y_vals_train, dtype=tf.float32), [-1,1]), batch_size=50)

# model
sess = tf.Session()
model = SVM(dataset=dataset, config=config, parameters=Parameters())
sess.run(tf.global_variables_initializer())
model.training(session=sess)
print(sess.run([model.weights, model.bias]))
# print(sess.run(model.accuracy))


#output vizualisation
weights, bias=sess.run([model.weights, model.bias])
slope = -weights[1]/weights[0]
y_intercept = bias[0]/weights[0]
x1_vals = [d[1] for d in x_vals]
best_fit = []
for i in x1_vals:
    best_fit.append(slope*i+y_intercept)
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator',linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()




