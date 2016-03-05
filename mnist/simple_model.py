import os
import shutil

import input_data
import tensorflow as tf

tmp_filename = "/tmp/mnist_simple_model_logs"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)


w_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

merged = tf.merge_all_summaries()
shutil.rmtree(tmp_filename)
writer = tf.train.SummaryWriter(tmp_filename, sess.graph_def)

for i in range(1000000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        result = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        summary_str = result[0]
        acc = result[1]
        writer.add_summary(summary_str, i)
        print("Accuracy at step %s: %s" % (i, acc))


print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
