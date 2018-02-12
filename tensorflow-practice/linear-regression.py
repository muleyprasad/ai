import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#create some test data and simulate results
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T) + b_real + noise

print(len(x_data))
print(len(y_data[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = x_data[:,0]
x2 = x_data[:,1]
x3 = x_data[:,2]
ax.scatter3D(x1, x2, x3, c=x3, cmap='Greens');

plt.show()

#linear regression, determine weight and bias

NUM_STEPS = 10
LEARNING_RATE = 0.5

g = tf.Graph()
wb_ = []

with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
    b = tf.Variable(0,dtype=tf.float32,name='bias')

    y_pred = tf.matmul(w,tf.transpose(x)) + b

    loss = tf.reduce_mean(tf.square(y_true-y_pred))

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