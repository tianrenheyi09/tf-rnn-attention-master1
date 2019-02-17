import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
def attention( inputs, output_size):
    """
    desc: create attention mechanism
    args:
      inputs: input which is sentence or document level output from bidirectional rnn layer
      output_size: specify the dimensions of the output
    returns:
      output from attention distribution
    """

    with tf.variable_scope("attention"):
      attention_context_vector_uw = tf.Variable(tf.truncated_normal([output_size]), name='u_context')
      input_projection_u = layers.fully_connected(inputs,
                                                  output_size,
                                                  #trainable=self.is_training,
                                                  activation_fn=tf.tanh)
      vector_attn = tf.reduce_sum(tf.multiply(input_projection_u, attention_context_vector_uw), axis=2, keep_dims=True)
      attention_weights = tf.nn.softmax(vector_attn, dim=1)
      weighted_projection = tf.multiply(input_projection_u, attention_weights)
      outputs = tf.reduce_sum(weighted_projection, axis=1)
      return outputs

x = np.arange(480).reshape(10,8,6)

mean1 = tf.reduce_mean(x, axis=0)
mean2 = tf.reduce_mean(x,axis=0,keep_dims=True)

output = attention(x,3)

with tf.Session() as sess:
    print(x.shape)
    print(output.shape)

    # print(sess.run(tf.reduce_mean(x, axis=0)))
    # print(sess.run(tf.reduce_mean(x,axis=0,keep_dims=True)))
    # print(sess.run(tf.reduce_mean(x,axis=1)))