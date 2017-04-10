# encoding: utf8
'''
Distributed Tensorflow example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python testDist.py --job_name="ps" --task_index=0 --model_dir=0
pc-02$ python testDist.py --job_name="worker" --task_index=0  --model_dir=1
pc-03$ python testDist.py --job_name="worker" --task_index=1  --model_dir=2
pc-04$ python testDist.py --job_name="worker" --task_index=2  --model_dir=3

More details here: ischlag.github.io
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np

from tensorflow.python.client import timeline

# cluster specification
parameter_servers = ["search-bridge-deep2:2222"]
workers = [ "search-bridge-deep5:2223", 
            "search-bridge-deep6:2224"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string('model_dir', '', 'Directory for model root.')
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.001
training_epochs = 100
logs_path = "./logs"


def getData():
    N = 100000 / len(workers) # class마다 점의 갯수
    D = 2 # point의 dimension
    K = 3 # class 갯수

    X = np.zeros((N*K, D)) # data matrix (각 row가 data point)
    y = np.zeros((N*K, K), dtype='uint8') # class label들

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta

        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] # concatenation along the second axis.
        y[ix, j] = 1

    
    indices = np.linspace(0, N*K-1, N*K, dtype=int)
    np.random.shuffle(indices)

    return X[indices,:], y[indices,:], K

X, Y, K = getData()


if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):

    # count the number of updates
    global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)

    # input images
    with tf.name_scope('input'):
      # None -> batch size can be any size, 784 -> flattened mnist image
      x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, K], name="y-input")

    # model parameters will change during training so we use tf.Variable
    tf.set_random_seed(1)
    with tf.device("/cpu:0"):
      with tf.name_scope("weights"):
        W1 = tf.Variable(tf.random_normal([2, 10000]))
        W3 = tf.Variable(tf.random_normal([10000, 5000]))
        W2 = tf.Variable(tf.random_normal([5000, K]))

      # bias
      with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([10000]))
        b3 = tf.Variable(tf.zeros([5000]))
        b2 = tf.Variable(tf.zeros([K]))

    with tf.name_scope("softmax"):
      # implement model
      with tf.device("/gpu:0"):
        # y is our prediction
        z2 = tf.add(tf.matmul(x,W1),b1)
      with tf.device("/gpu:1"):
        a2 = tf.nn.relu(z2)
      with tf.device("/gpu:2"):
        z3 = tf.add(tf.matmul(a2,W3),b3)
      with tf.device("/gpu:3"):
        a3 = tf.nn.relu(z3)
      with tf.device("/gpu:4"):
        z4 = tf.add(tf.matmul(a3,W2),b2)
      with tf.device("/gpu:5"):
        y  = tf.nn.softmax(z4)

    # specify cost function
    with tf.name_scope('cross_entropy'):
      # this is our cost
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # specify optimizer
    with tf.name_scope('train'):
      # optimizer is an "operation" which we can execute in a session
      grad_op = tf.train.GradientDescentOptimizer(learning_rate)
      '''
      rep_op = tf.train.SyncReplicasOptimizer(grad_op, 
                                          replicas_to_aggregate=len(workers),
                                          replica_id=FLAGS.task_index, 
                                          total_num_replicas=len(workers),
                                          use_locking=True
                                          )
      train_op = rep_op.minimize(cross_entropy, global_step=global_step)
      '''
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)
      
    '''
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()
    '''

    with tf.name_scope('Accuracy'):
      # accuracy
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.summary.merge_all()
    init_op = tf.initialize_all_variables()
    print("Variables initialized ...")

  saver = tf.train.Saver()
  sv = tf.train.Supervisor(logdir=FLAGS.model_dir + "%d" % FLAGS.task_index,
                           is_chief=(FLAGS.task_index == 0),
                           saver=saver,
                           init_op=tf.global_variables_initializer(),
                           summary_op=None,
                           save_model_secs=3600,
                           global_step=global_step)

  begin_time = time.time()
  frequency = 100
  
  with sv.prepare_or_wait_for_session(server.target,
            config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # create log writer object (this will log on every machine)
    writer = tf.summary.FileWriter(FLAGS.model_dir, graph=tf.get_default_graph())
        
    # perform training cycles
    start_time = time.time()
    for epoch in range(2):#range(training_epochs):

      # number of batches in one epoch
      batch_count = int(X.shape[0]/batch_size)

      count = 0
      for i in range(batch_count):
        batch_x, batch_y = X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]
        
        # perform the operations we defined earlier on batch
        _, cost, summary, step = sess.run(
                        [train_op, cross_entropy, summary_op, global_step], 
                        feed_dict={x: batch_x, y_: batch_y},
                        options=run_options, run_metadata=run_metadata)
        writer.add_summary(summary, step)

        count += 1
        if count % frequency == 0 or i+1 == batch_count:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print("Step: %d," % (step+1), 
                " Epoch: %2d," % (epoch+1), 
                " Batch: %3d of %3d," % (i+1, batch_count), 
                " Cost: %.4f," % cost, 
                " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
          count = 0

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    print("Final Cost: %.4f" % cost)

  sv.stop()
  print("done")