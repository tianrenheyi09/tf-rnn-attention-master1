import tensorflow as tf
import tensorflow.contrib.layers as layers

def attention1(inputs, output_size):
    """
    desc: create attention mechanism
    args:
      inputs: input which is sentence or document level output from bidirectional rnn layer
      output_size: specify the dimensions of the output
    returns:
      output from attention distribution
    """

    with tf.variable_scope("attention"):
      attention_context_vector_uw = tf.get_variable(name="attention_context_vector",
                                                    shape=[output_size],
                                                    #trainable=self.is_training,
                                                    initializer=layers.xavier_initializer(),
                                                    dtype=tf.float32)
      input_projection_u = layers.fully_connected(inputs,
                                                  output_size,
                                                  #trainable=self.is_training,
                                                  activation_fn=tf.tanh)
      print("input_projection_u size: ",input_projection_u.shape)
      uu = tf.multiply(input_projection_u, attention_context_vector_uw)
      print("uu tf.matmly size: ",uu.shape)

      vector_attn = tf.reduce_sum(tf.multiply(input_projection_u, attention_context_vector_uw), axis=2, keep_dims=True)
      print("vector_attn sizzze: ",vector_attn.shape)
      attention_weights = tf.nn.softmax(vector_attn, dim=1)
      print("attention_weights size: ",attention_weights.shape)
      weighted_projection = tf.multiply(input_projection_u, attention_weights)
      print(" weighted_projection  size: ", weighted_projection.shape)
      outputs = tf.reduce_sum(weighted_projection, axis=1)

      return outputs