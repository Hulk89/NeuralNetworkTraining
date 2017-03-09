import tensorflow as tf
import numpy as np

A = [[1,2,3],[4,5,6]]
A = np.array(A)

B = tf.placeholder(tf.int32, [2, 3], name='B')

sess = tf.Session()
sess.run(B, feed_dict={B:A})

C = tf.unstack(B)
print( sess.run(C, feed_dict={B:A}) )

