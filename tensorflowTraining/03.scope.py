import tensorflow as tf


with tf.variable_scope('scope1'):
    tf.get_variable('A',
                    [1, 1],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))

with tf.variable_scope('scope1'):
    tf.get_variable('B',
                    [1, 1],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))

with tf.Session() as sess:
    
    file_writer = tf.summary.FileWriter('/test', sess.graph)