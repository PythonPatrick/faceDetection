import tensorflow as tf
import numpy as np

x = tf.Variable(0, name='x')

def func(x):
    return x+1

def func2(x):
    return x*2

result=x.assign(func(x))
result2=x.assign(func2(result))

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        a,b=session.run([result,result2])
        print(session.run(x))