# encoding: utf8
import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(5.0)
c = a * b

A = tf.get_variable("A", [20, 1], initializer=tf.random_normal_initializer())

c_summary = tf.summary.scalar("point", c)
a_summary = tf.summary.histogram("hist", A)
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("./summary")

with tf.Session() as sess:
    tf.global_variables_initializer().run()    

    result = sess.run(merged)

    writer.add_summary(result)

    writer.flush()