from __future__ import division, print_function, absolute_import

import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from sklearn import cross_validation
import data_utils

# flags
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.flags.DEFINE_integer("epochs", 20, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 50, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("num_hidden_nodes", 200, "Number of hidden nodes inside A")
tf.flags.DEFINE_integer("display_step", 10, "display results every 10 time steps")
# hyper-parameters
FLAGS = tf.flags.FLAGS

# print flags info
orig_stdout = sys.stdout
timestamp = str(int(time.time()))
folder_name = 'essay_set_{}'.format(timestamp)
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", folder_name))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Parameters
learning_rate = FLAGS.learning_rate
#training_iters = 100000
batch_size = FLAGS.batch_size
display_step = FLAGS.display_step
epochs = FLAGS.epochs
# Network Parameters
n_input = FLAGS.embedding_size
#n_steps = ? # timesteps = num of words in article, cannot defined here
n_hidden = FLAGS.num_hidden_nodes # hidden layer num of features
n_classes = 2 # total classes (0-1 digits)

#deal with input data
training_path = 'training_set_rel3.tsv'     #put name of training file here
essay_list, label = data_utils.load_training_data(training_path)

#for i in range(len(label)):
#    if label[i] > 4:
#        label[i] = 1
#    else:
#        label[i] = 0

# load glove
word_idx, word2vec = data_utils.load_glove(n_input)

vocab_size = len(word_idx) + 1
# stat info on data set

sent_size_list = map(len, [essay for essay in essay_list])
max_sent_size = max(sent_size_list)
mean_sent_size = int(np.mean(map(len, [essay for essay in essay_list])))

print ('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))
with open(out_dir+'/params', 'a') as f:
    f.write('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))
#input x
E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)
#input y : convert score to one hot encoding
S = np.eye(n_classes)[label]
# split the data on the fly
trainE, testE, train_scores, test_scores, train_sent_sizes, test_sent_sizes \
    = cross_validation.train_test_split(E, S, sent_size_list, test_size=.2)

trainE, evalE, train_scores, eval_scores, train_sent_sizes, eval_sent_sizes \
    = cross_validation.train_test_split(trainE, train_scores, train_sent_sizes, test_size=.1)

# data size
n_train = len(trainE)
n_test = len(testE)
n_eval = len(evalE)

print ('The size of training data: {}'.format(n_train))
print ('The size of testing data: {}'.format(n_test))
print ('The size of evaluation data: {}'.format(n_eval))

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

# tf input
n_steps = max_sent_size
print ("n_steps = max_sent_size = ")
print (n_steps)
#x = tf.placeholder("float", [None, n_steps, n_input])
x = tf.placeholder(tf.int32, [None, n_steps], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")

#do word embedding
#x = tf.nn.embedding_lookup(word2vec, x)
temp_placeholder = tf.placeholder(tf.float32, [vocab_size, n_input], name="w_placeholder")
temp = tf.Variable(temp_placeholder, trainable=False)
x_embedded = tf.nn.embedding_lookup(temp, x)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x_embedded, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init, feed_dict={temp_placeholder: word2vec})
    step = 1
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    for i in range(1, epochs+1):
        print("epoch " + str(i)+":")
        for start, end in batches:
            batch_x = trainE[start:end]
            batch_y = train_scores[start:end]
            # Run optimization op (backprop)
            #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
            step += 1

    print ("Optimization Finished!")

    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testE, y: test_scores}))

