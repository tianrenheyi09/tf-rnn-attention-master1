
import tensorflow as tf
import numpy as np
import copy

class TextRCNN(object):

    def __init__(self,model_type,sequence_length,num_calsses,vocab_size,embedding_size,batch_size,l2_reg_lambda=0.5):

        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None,num_calsses],name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32,name="learing_rate")

        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1,1),name="W",trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)

        with tf.name_scope('rcnn'):

            self.initializer = tf.random_normal_initializer(stddev=0.1)
            self.left_side_first_word = tf.get_variable("left_side_first_word",shape=[batch_size,embedding_size],initializer=self.initializer)
            self.right_side_last_word = tf.get_variable("right_side_last_word",shape=[batch_size,embedding_size],initializer=self.initializer)
            self.W_l = tf.get_variable("W_l",shape=[batch_size,embedding_size],initializer=self.initializer)
            self.W_r = tf.get_variable("W_r",shape=[batch_size,embedding_size],initializer=self.initializer)
            self.W_sl = tf.get_variable("W_sl",shape=[batch_size,embedding_size],initializer=self.initializer)
            self.W_sr = tf.get_variable("W_sr",shape=[batch_size,embedding_size],initializer=self.initializer)
            ##rnn-cnnn
            def get_context_left(context_left,embedding_previous):
                left_c = tf.matmul(context_left,self.W_l)
                left_e = tf.matmul(embedding_previous,self.W_sl)
                left_h = left_c+left_e
                context_left = tf.nn.relu(left_h,name="relu")
                return context_left
            def get_context_right(context_right,embedding_afterward):
                right_c = tf.matmul(context_right,self.W_r)
                right_e = tf.matmul(embedding_afterward,self.W_sr)
                right_h = right_c+right_e
                context_right = tf.nn.relu(right_h,name="relu")

                return context_right

            embedded_words_split = tf.split(self.embedded_chars,sequence_length,axis=1)
            embedded_words_squeezed = [tf.squeeze(x,axis=1) for x in embedded_words_split]
            embedding_previous = self.left_side_first_word
            context_left_previous = tf.zeros((batch_size,embedding_size))
            context_left_list = []
            for i,current_embedding_word in enumerate(embedded_words_squeezed):
                context_left = get_context_left(context_left_previous,embedding_previous)
                context_left_list.append(context_left)
                embedding_previous = current_embedding_word
                context_left_previous = context_left

            embedded_words_squeezed2 = copy.copy(embedded_words_squeezed)
            embedded_words_squeezed2.reverse()
            embedding_afterward = self.right_side_last_word
            context_right_afterward = tf.zeros((batch_size,embedding_size))
            context_right_list  = []
            for j,current_embedding_word in enumerate(embedded_words_squeezed2):
                context_right = get_context_right(context_right_afterward,embedding_afterward)
                context_right_list.append(context_right)
                embedding_afterward = current_embedding_word
                context_right_afterward = context_right

            output_list = []
            for index ,current_embedding_word in enumerate(embedded_words_squeezed):
                representation = tf.concat([context_left_list[index],current_embedding_word,context_right_list[index]],axis=1)
                output_list.append(representation)

            outputs = tf.stack(output_list,axis=1)
            self.output = tf.reduce_max(outputs,axis=1)

        with tf.name_scope("dropput"):
            self.rnn_drop = tf.nn.dropout(self.output,self.dropout_keep_prob)

        with tf.name_scope("output"):
                W = tf.get_variable("W",shape=[embedding_size*3,num_calsses],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1,shape=[num_calsses]),name="b")
                l2_loss +=tf.nn.l2_loss(W)
                l2_loss +=tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.rnn_drop,W,b,name="scores")
                if model_type == 'clf':
                    self.predictions = tf.argmax(self.scores,1,name="predictions")
                elif model_type == 'reg':
                    self.predictions = tf.reduce_max(self.scores,1,name="predictions")
                    self.predictions = tf.expand_dims(self.predictions,-1)
        with tf.name_scope("loss"):
            if model_type == "clf":
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            elif model_type == "reg":
                losses = tf.sqrt(tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_y))
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            if model_type == "clf":
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            elif model_type == "reg":
                self.accuracy = tf.constant(0.0, name="accuracy")































