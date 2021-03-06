#!/usr/bin/python
"""
Toy example of attention layer use

Train RNN (GRU) on IMDB dataset (binary classification)
Learning and hyper-parameters were not tuned; script serves as an example
"""
from __future__ import print_function, division

import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm

from attention import attention
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 32#100
HIDDEN_SIZE = 100#150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 256
NUM_EPOCHS = 10  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
MODEL_PATH = './model'

# Load the data set
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
#
# # Sequences pre-processing
# vocabulary_size = get_vocabulary_size(X_train)
# X_test = fit_in_vocabulary(X_test, vocabulary_size)
# X_train = zero_pad(X_train, SEQUENCE_LENGTH)
# X_test = zero_pad(X_test, SEQUENCE_LENGTH)

############-----------------------载入imdb数据
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import to_categorical
# max_features = 10000
# maxlen = 500
# batch_size = 32
print("load data")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
print(len(X_train),"train_sequence")
print(len(X_test),"test sequence")

print("padding _squence")
X_train = sequence.pad_sequences(X_train,maxlen=SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test,maxlen=SEQUENCE_LENGTH)
# print(input_train.shape,"input_trian_shape")
# print(input_test.shape,"input_test_shape")

#####-----------y_train转化为onr_hot
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

##########----------字典的大小-----
vocabulary_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])])+1
print("vocab_size,--字典大小   ",vocabulary_size)

num_classes = 2
# Different placeholders
with tf.name_scope('Inputs'):
    batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None,num_classes], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
# rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
#                         inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)

# rnn_outputs,_ = bi_rnn(GRUCell(HIDDEN_SIZE),GRUCell(HIDDEN_SIZE),batch_embedded,dtype=tf.float32)
rnn_outputs= tf.reduce_mean(batch_embedded,axis=1)
tf.summary.histogram('RNN_outputs', rnn_outputs)

# # Attention layer
# with tf.name_scope('Attention_layer'):
#     attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
#     tf.summary.histogram('alphas', alphas)

# Dropout
drop = tf.nn.dropout(rnn_outputs, keep_prob_ph)



with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([EMBEDDING_DIM,num_classes],stddev=0.1))
    b = tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
    y_hat = tf.nn.xw_plus_b(drop,W,b)
    y_hat1 = tf.argmax(y_hat,1,name="y_hat1")
    tf.summary.histogram('W',W)



# Fully connected layer
# with tf.name_scope('Fully_connected_layer'):
#     W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
#     b = tf.Variable(tf.constant(0., shape=[1]))
#     y_hat = tf.nn.xw_plus_b(drop, W, b)
#     y_hat = tf.squeeze(y_hat)
#     tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    correct_predictions = tf.equal(y_hat1, tf.argmax(target_ph, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat1)), target_ph), tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# from tensor_textfast import TextFast
#
# nn = TextFast(
#     model_type='clf',
#     sequence_length=SEQUENCE_LENGTH,
#     num_classes=2,
#     vocab_size=vocabulary_size,
#     embedding_size=EMBEDDING_DIM,
#     l2_reg_lambda=0.5
# )
#
# tf.summary.scalar('loss',nn.loss)
# tf.summary.scalar("accuracy",nn.accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()


import  matplotlib.pyplot as plt

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(train_batch_generator)
                # seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                               target_ph: y_batch,
                                                               # seq_len_ph: seq_len,
                                                               keep_prob_ph: 0.8})
                accuracy_train += acc
                # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                loss_train = loss_train+loss_tr
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches
            loss_train /=num_batches

            # Testing
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(test_batch_generator)
                # seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={batch_ph: x_batch,
                                                                    target_ph: y_batch,
                                                                    # seq_len_ph: seq_len,
                                                                    keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        train_writer.close()
        test_writer.close()
        saver.save(sess, MODEL_PATH)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")



