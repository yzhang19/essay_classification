from __future__ import division, print_function, absolute_import
#from sklearn.model_selection import GroupKFold
import os
import sys
import time
#test
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from sklearn import cross_validation
import data_utils

# flags
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("epochs", 50, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("batch_size", 300, "Batch size for training.")
tf.flags.DEFINE_integer("num_hidden_nodes", 200, "Number of hidden nodes inside A")
tf.flags.DEFINE_integer("display_step", 2, "display results every 10 time steps")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "Number of layers in multilayers RNN")
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
training_path = '200_clean_graded_data.csv'     #put name of training file here
essay_list, label, problem_id, count_one, question_list = data_utils.load_open_response_data(training_path)

#majorty class

print ('majorty class accuracy is : \n', count_one/len(label))
# load glove
word_idx, word2vec = data_utils.load_glove(n_input)

vocab_size = len(word_idx) + 1
# stat info on data set

sent_size_list = map(len, [essay for essay in essay_list])
max_sent_size = max(sent_size_list)
mean_sent_size = int(np.mean(map(len, [essay for essay in essay_list])))

question_sent_size_list = map(len, [question for question in question_list])
question_max_sent_size = max(question_sent_size_list)
question_mean_sent_size = int(np.mean(map(len, [question for question in question_list])))

print ('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))
#exit()
with open(out_dir+'/params', 'a') as f:
    f.write('max sentence size: {} \nmean sentence size: {}\nquestion max sentence size: {}\n'
            'question mean sentence size:{}\n'.format(max_sent_size, mean_sent_size, question_max_sent_size, question_mean_sent_size))
    f.write('majorty class accuracy is : {}\n'.format(count_one/len(label)))

#input x
E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)

#input question
Q = data_utils.vectorize_data(question_list, word_idx, question_max_sent_size)
#input y : convert score to one hot encoding
S = np.eye(n_classes)[label]
# split the data on the fly
#do the cross validation: test problem is not included in training set
#gkf = GroupKFold(n_splits=5)
#for train_index, test_index in gkf.split(E, S, groups = problem_id):
#    trainE = [E[i] for i in train_index]
#    testE = [E[i] for i in test_index]
#    train_scores = [S[i] for i in train_index]
#    test_scores = [S[i] for i in test_index]
#    break

#random split the data set to training set and testing set
trainE, testE, train_scores, test_scores, trainQ, testQ, train_sent_sizes, test_sent_sizes, train_Q_sent_sizes, test_Q_sent_sizes \
    = cross_validation.train_test_split(E, S, Q, sent_size_list, question_sent_size_list, test_size=.2)

#trainE, evalE, train_scores, eval_scores, train_sent_sizes, eval_sent_sizes \
 #   = cross_validation.train_test_split(trainE, train_scores, train_sent_sizes, test_size=.1)

# data size
n_train = len(trainE)
n_test = len(testE)
#n_eval = len(evalE)

print ('The size of training data: {}'.format(n_train))
print ('The size of testing data: {}'.format(n_test))
#print ('The size of evaluation data: {}'.format(n_eval))

batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

batches_test = zip(range(0, n_test-batch_size, batch_size), range(batch_size, n_test, batch_size))
batches_test = [(start_test, end_test) for start_test, end_test in batches_test]

# tf input
n_steps = max_sent_size
print ("n_steps = max_sent_size = ")
print (n_steps)
#x = tf.placeholder("float", [None, n_steps, n_input])
x = tf.placeholder(tf.int32, [None, n_steps], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")
p = tf.placeholder(tf.int32, [None, question_max_sent_size], name="p")
#do word embedding
#x = tf.nn.embedding_lookup(word2vec, x)
#temp is wordembedding
temp_placeholder = tf.placeholder(tf.float32, [vocab_size, n_input], name="w_placeholder")
temp = tf.Variable(temp_placeholder, trainable=False)
x_embedded = tf.nn.embedding_lookup(temp, x)
p_embedded = tf.nn.embedding_lookup(temp, p)
#xp = tf.cross (x_embedded, p_embedded, name = None)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases, n_steps):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    hidden_layers = []
    for i in range(FLAGS.num_rnn_layers):
        # Define a lstm cell with tensorflow
        hidden1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        hidden_layers.append(hidden1)

    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)
    lstm_cell = rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.add_n(outputs)/len(outputs)

with tf.variable_scope('rnn') as scope:
    x_output = RNN(x_embedded, weights, biases, n_steps)
    scope.reuse_variables()
    p_output = RNN(p_embedded, weights, biases, question_max_sent_size)

output = tf.concat(1, [x_output, p_output])
pred = tf.matmul(output, weights['out']) + biases['out']
pred_prob = tf.nn.softmax(pred)[:, 1]
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

# Evaluate model
pred_value = tf.argmax(pred,1)
actual_value = tf.argmax(y,1)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init, feed_dict={temp_placeholder: word2vec})
    step = 1
    test_count = 1
    # Keep training until reach max iterations
    #while step * batch_size < training_iters:
    for i in range(1, epochs+1):
        print("epoch " + str(i)+":")
        for start, end in batches:
            batch_x = trainE[start:end]
            batch_y = train_scores[start:end]
            batch_p = trainQ[start:end]
            # Run optimization op (backprop)
            #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            sess.run(optimizer, feed_dict={x: batch_x, p:batch_p, y: batch_y})
            # Calculate batch accuracy
            acc, pred_score, acutal_score = sess.run([accuracy, pred_value, actual_value], feed_dict={x: batch_x, p:batch_p, y: batch_y})

            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, p: batch_p, y: batch_y})
            #print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
            #       "{:.6f}".format(loss) + ", Training Accuracy= " +
            #       "{:.5f}".format(acc))

        if i % FLAGS.display_step == 0:
            count = 0
            testacc = 0
            with open(out_dir+'/results'+str(test_count), 'a') as f:
                f.write('pred\tactual\n')
            for start_test, end_test in batches_test:
                batch_testx = testE[start_test:end_test]
                batch_testy = test_scores[start_test:end_test]
                batch_testp = testQ[start_test:end_test]
                count += 1
                #testacc = testacc + sess.run(accuracy, feed_dict={x: batch_testx, y: batch_testy});
                temp_testacc, test_prob, pred_score, actual_score = sess.run([accuracy, pred_prob, pred_value, actual_value], feed_dict={x: batch_testx, p:batch_p, y: batch_testy})
                testacc = testacc + temp_testacc
                with open(out_dir+'/probs'+str(test_count), 'a') as f:
                    for i in range(len(test_prob)):
                        f.write('{}\n'.format(test_prob[i]))
                with open(out_dir+'/results'+str(test_count), 'a') as f:
                    for i in range(len(pred_score)):
                        f.write('{}\t{}\n'.format(pred_score[i], actual_score[i]))
                        #with open(out_dir+'/results', 'a') as f:
                        #    f.write('{}\t{}\n'.format(pred_score[i], actual_score))
            with open(out_dir+'/params', 'a') as f:
                f.write("Testing Accuracy:{}\n".format(testacc/count))
            print ("Testing Accuracy:\n", testacc/count)
            test_count += 1
