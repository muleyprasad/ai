import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1/(1+np.exp(-x))

#create some test data and simulate results
x_data = np.random.randn(20000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

y_data_before_noise = sigmoid(np.matmul(w_real,x_data.T) + b_real)
y_data = np.random.binomial(1,y_data_before_noise)

#linear regression, determine weight and bias

NUM_STEPS = 50
LEARNING_RATE = 0.5

g = tf.Graph()
wb_ = []

with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
    b = tf.Variable(0,dtype=tf.float32,name='bias')

    y_pred = tf.matmul(w,tf.transpose(x)) + b

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)
    
    # now run in a session
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for step in range(NUM_STEPS):
            session.run(train,{ x: x_data,y_true: y_data})
            print(step, session.run([w,b]))

print('Matches w_real & b_real :', w_real, b_real)