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
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator,batch_iter,batch_iter_my
import datetime
from sklearn import metrics
import time
import pandas as pd
import os
from tensor_textfast import TextFast
from tensor_textdnn import TextDNN
from tensor_textrnn import TextRNN
from tensor_textrcnn import TextRCNN
from tensor_textcnn import TextCNN
from text_han import TextHAN
from tensor_birnn import TextBiRNN
# from han import HAN
from HAN_model import HAN
from tensor_birr_atten import TextBiRNN_Atten

#####----Data loading params
model_type = "clf" #"the type of model ,classify or regression(defalut=clf)"
using_nn_type = "text_cnn" #the type of network(default:textcnn)
dev_sample_percentage = 0.1 # "Percentage of the training data to use for validation"
train_data_file =  "./data/cutclean_label_corpus10000.txt" #"Data source for the positive data."
train_label_data_file = "" #"Data source for the label data."
w2v_file =  "./data/vectors.bin"  #"w2v_file path"

#####----model  paramns
filter_sizes = "2,3,4"  # "Comma-separated filter sizes (default: '3,4,5')"
num_filters = 64  #"Number of filters per filter size (default: 128)"
# dropout_keep_prob = 0.5  #"Dropout keep probability (default: 0.5)"
l2_reg_lambda = 0.0  #"L2 regularization lambda (default: 0.0)"
hidden_layers  = 2  # "Number of hidden layers(default:2)"
rnn_size = 100  #"num of units rnn_size(default:3)"
num_rnn_layers  = 3  #"number of rnn layers(default:3)"
####--------------imdb   params
num_words = 10000
index_from = 0
sequence_length = 250
embedding_dim = 100#100 #"Dimensionality of character embedding (default: 128)"
hidden_size = 100#150#"Number of hiddern layer units(defalut:128)"
attention_size = 50
keep_prob = 0.8


delte = 0.5
model_path='./model'

######----------training parmas
batch_size  = 256 #"Batch Size (default: 64)"
num_epochs = 3# "Number of training epochs (default: 200)"
# evaluate_every = 50  #"Evaluate model on dev set after this many steps (default: 100)"
# checkpoint_every = 100  #"Save model after this many steps (default: 100)"
# num_checkpoints = 5  ##"Number of checkpoints to store (default: 5)"




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

print("load data")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(len(X_train),"train_sequence")
print(len(X_test),"test sequence")

print("padding _squence")
X_train = sequence.pad_sequences(X_train,maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test,maxlen=sequence_length)
# print(input_train.shape,"input_trian_shape")
# print(input_test.shape,"input_test_shape")
label_train = y_train.copy()
label_test = y_test.copy()
#####-----------y_train转化为onr_hot
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


##########----------字典的大小-----
vocabulary_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])])+1
print("vocab_size,--字典大小   ",vocabulary_size)




# nn = TextFast(
#     model_type='clf',
#     sequence_length=sequence_length,
#     num_classes=2,
#     vocab_size=vocabulary_size,
#     embedding_size=embedding_dim,
#     l2_reg_lambda=0.5
# )
nn = TextBiRNN_Atten(
    model_type='clf',
    sequence_length=sequence_length,
    num_classes=y_train.shape[1],
    vocab_size=vocabulary_size,
    embedding_size=embedding_dim,
    rnn_size=rnn_size,
    attention_size=rnn_size,
    l2_reg_lambda=0.5
)
# nn = TextRNN(
#     model_type='clf',
#     sequence_length=X_train .shape[1],
#     num_classes=y_train.shape[1],
#     vocab_size=vocabulary_size,
#     embedding_size=embedding_dim,
#     rnn_size=128,
#     num_layers=2,
#     l2_reg_lambda=0.5
# )
# nn = TextCNN(
#     model_type='clf',
#     sequence_length=X_train.shape[1],
#     num_classes=y_train.shape[1],
#     vocab_size=vocabulary_size,
#     embedding_size=embedding_dim,
#     filter_sizes=list(map(int,filter_sizes.split(","))),
#     num_filters=num_filters,
#     l2_reg_lambda=0.5
# )
# nn = TextHAN(
#     model_type='clf',
#     sequence_length=X_train.shape[1],
#     num_sentences=5,
#     num_classes=y_train.shape[1],
#     vocab_size=vocabulary_size,
#     embedding_size=embedding_dim,
#     hidden_size=rnn_size,
#     batch_size=batch_size,
#     l2_reg_lambda=0.5
# )
# nn = TextBiRNN(
#     model_type='clf',
#     sequence_length=X_train.shape[1],
#     num_classes=y_train.shape[1],
#     vocab_size=vocabulary_size,
#     embedding_size=embedding_dim,
#     rnn_size=rnn_size,
#     num_layers=2,
#     l2_reg_lambda=0.5
# )
# nn = HAN(
#     vocab_size=vocabulary_size,
#     num_classes=y_train.shape[1],
#     embedding_size=embedding_dim,
#     hidden_size=rnn_size
# )


########--------使用SGD梯度下降最小loss
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(nn.loss)

tf.summary.scalar('loss',nn.loss)
tf.summary.scalar("accuracy",nn.accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, batch_size)
test_batch_generator = batch_generator(X_test, y_test,batch_size)


train_writer = tf.summary.FileWriter('./logdir/train', nn.accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', nn.accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

batches_tr = batch_iter_my(list(zip(X_train,y_train)),batch_size,True)
num_batches = int((len(X_test) - 1) / batch_size) + 1
for index,batchs in enumerate(batches_tr):
    x_batch,y_batch = zip(*batchs)
    print(x_batch.shape)



if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for epoch in range(num_epochs):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            batches_tr = batch_iter_my(list(zip(X_train,y_train)),batch_size,True)
            num_batches = int((len(X_test) - 1) / batch_size) + 1
            for index,batchs in enumerate(batches_tr):
                x_batch,y_batch = zip(*batchs)
                loss_tr, acc, _, summary = sess.run([nn.loss, nn.accuracy, optimizer, merged],
                                                    feed_dict={nn.input_x: x_batch,
                                                               nn.input_y: y_batch,
                                                               # seq_len_ph: seq_len,
                                                               nn.dropout_keep_prob: 0.8})

                accuracy_train += acc
                # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                loss_train = loss_train + loss_tr
                train_writer.add_summary(summary, index + num_batches * epoch)
                print("index:{},acc:{:.3f},loss:{:.3f}".format(index, acc, loss_tr))

            accuracy_train /= num_batches
            loss_train /= num_batches

            # Testing
            batch_dev = batch_iter_my(list(zip(X_test,y_test)),batch_size,True)
            for index,batches in enumerate(batch_dev):
                x_batch,y_batch = zip(*batches)
                loss_test_batch, acc, summary = sess.run([nn.loss, nn.accuracy, merged],
                                                         feed_dict={nn.input_x: x_batch,
                                                                    nn.input_y: y_batch,
                                                                    # seq_len_ph: seq_len,
                                                                    nn.dropout_keep_prob: 1.0})

                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, index + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
        ###########---------------eval  data
        def softmax(x):
            if(x.ndim ==1):
                x = x.reshape(1,-1)
            max_x = np.max(x,axis=1).reshape((-1,1))
            exp_x = np.exp(x-max_x)
            return exp_x/np.sum(exp_x,axis=1).reshape((-1,1))

        # batches = batch_iter(list(X_test),batch_size,1,shuffle=False)
        batches = batch_iter_my(list(X_test),batch_size,False)
        all_predictions = []
        all_probabilities = None
        ####------all_predictions值为0,1,但是all_probabilities值为两列，第一列为0的概率值，第二列为1的概率值
        for index,x_test_batch in enumerate(batches):
            batch_predictions = sess.run([nn.predictions,nn.scores],feed_dict={nn.input_x:x_test_batch,nn.dropout_keep_prob:1})
            all_predictions = np.concatenate([all_predictions,batch_predictions[0]])
            probabilities = softmax(batch_predictions[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities,probabilities])
            else:
                all_probabilities = probabilities
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}".format(time_str,(index+1)*batch_size))

        if y_test is not None:
            y_test_eval = label_test####y_test转化为一维
            correct_predictions = float(sum(all_predictions == y_test_eval))
            print("total numberg of test examples: {}".format(len(y_test_eval)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test_eval))))
            print(metrics.classification_report(y_test_eval,all_predictions))
            print(metrics.confusion_matrix(y_test_eval,all_predictions))

        #####---------------save  results----
        predictions_read = np.column_stack(all_predictions)
        out_path = os.path.join("prdiction.csv")
        print("saving evalutation to {0}".format(out_path))

        predictions_readone = np.column_stack(([int(pre) for pre in all_predictions],
                                              ["{}".format(pro) for pro in all_probabilities]))
        predict_results = pd.DataFrame(predictions_readone,columns=['Label','Probabilities'])
        print("saving evalutatation to {0}".format(out_path))
        predict_results.to_csv(out_path,index=False)

        train_writer.close()
        test_writer.close()
        saver.save(sess, model_path)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")



