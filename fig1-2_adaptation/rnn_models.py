import numpy as np
import tensorflow as tf

def rnn_fig1(STIMULI, TIMESTEP):
  # inputs: stimuli time series
  # stimuli.shape=[N_CASES,time_points,Gin]
  # output: expression level dynamic as tf.tensor
  # output shape [N_CASES,time_points,Gs]
  
  N_CASES = STIMULI.shape[0]
  time_points = STIMULI.shape[1]
  Gin = STIMULI.shape[2]
  Gs = 2
  Nwidth = 16
  
  W1 = tf.Variable(tf.truncated_normal([Gin+Gs,Nwidth], stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([Nwidth], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([Nwidth,Nwidth], stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([Nwidth], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([Nwidth, Gs], stddev=0.1))
  b3 = tf.Variable(tf.truncated_normal([Gs], stddev=0.1))
  
  hidden_init = tf.Variable(tf.random_uniform([Gs-1]))
  output_init = tf.constant([0.4])
  x0 = tf.concat([output_init, hidden_init], axis=0) #[Gs]

  x1 = tf.tile(tf.reshape(x0,[1,Gs]), [N_CASES,1]) #[N_CASES,Gs]
  X1_list = []

  for t1 in range(time_points):
    h0 = tf.concat([STIMULI[:,t1,:], x1], axis=1) #[N_CASES,Gin+Gs]

    a1 = tf.tensordot(h0, W1, [[1],[0]])
    h1 = tf.nn.relu(a1+b1) #[N_CASES,Nwidth]

    a2 = tf.tensordot(h1, W2, [[1],[0]])
    h2 = tf.nn.relu(a2+b2) #[N_CASES,Nwidth]

    a3 = tf.tensordot(h2, W3, [[1],[0]])
    h3 = tf.nn.sigmoid(a3+b3) #[N_CASES,Gs]

    x1 = (1-TIMESTEP)*x1 + TIMESTEP*h3
    X1_list.append(x1)

  X1_  = tf.stack(X1_list, axis=1) #[N_CASES,time_points,Gs]
  return X1_, [W1, b1, W2, b2, W3, b3, hidden_init]


def rnn_fig2(STIMULI, LINKS, TIMESTEP):
  # inputs: stimuli time series
  # stimuli.shape=[N_CASES,time_points,Gin]
  # input_mask.shape=[Gin,Gs]
  # LINKS.shape=[Gin+Gs,Gs]
  # output: expression level dynamic as tf.tensor
  # output shape [N_CASES,time_points,Gs]

  N_CASES = STIMULI.shape[0]
  time_points = STIMULI.shape[1]
  Gin = STIMULI.shape[2]
  Gs = 2
  Nwidth = 16
  
  W1 = tf.Variable(tf.truncated_normal([Gs,Gin+Gs,Nwidth], stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([Gs,Nwidth], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([Gs,Nwidth,Nwidth], stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([Gs,Nwidth], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([Gs,Nwidth], stddev=0.1))
  b3 = tf.Variable(tf.truncated_normal([Gs], stddev=0.1))
  
  hidden_init = tf.Variable(tf.random_uniform([Gs-1]))
  output_init = tf.constant([0.4])
  x0 = tf.concat([output_init, hidden_init], axis=0) #[Gs]

  x1 = tf.tile(tf.reshape(x0,[1,Gs]), [N_CASES,1]) #[N_CASES,Gs]
  X1_list = []

  for t1 in range(time_points):
    a0 = tf.concat([STIMULI[:,t1,:], x1], axis=1) #[N_CASES,Gin+Gs]
    h0 = tf.tile(tf.reshape(a0,[N_CASES,1,Gin+Gs]), [1,Gs,1]) #[N_CASES,Gs,Gin+Gs]
    h0_= h0*tf.transpose(LINKS) #*Links #[N_CASES,Gs,Gin+Gs]

    a1 = tf.reduce_sum(tf.reshape(h0_,[N_CASES,Gs,Gin+Gs,1])*tf.reshape(W1,[1,Gs,Gin+Gs,Nwidth]), axis=2)
    h1 = tf.nn.relu(a1+b1) #[N_CASES,Gs,Nwidth]

    a2 = tf.reduce_sum(tf.reshape(h1,[N_CASES,Gs,Nwidth,1])*tf.reshape(W2,[1,Gs,Nwidth,Nwidth]), axis=2)
    h2 = tf.nn.relu(a2+b2) #[N_CASES,Gs,Nwidth]

    a3 = tf.reduce_sum(h2*tf.reshape(W3,[1,Gs,Nwidth]), axis=2)
    h3 = tf.nn.sigmoid(a3+b3) #[N_CASES,N1,Gs]

    x1 = (1-TIMESTEP)*x1 + TIMESTEP*h3
    X1_list.append(x1)

  X1_  = tf.stack(X1_list, axis=1) #[N_CASES,time_points,Gs]
  return X1_
  