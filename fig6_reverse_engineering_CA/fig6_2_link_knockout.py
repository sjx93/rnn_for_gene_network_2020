import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FLAGS = None
BATCH_ic = 1 # different initial conditions
Gs = 10 # num of dynamic dims
Gin= Gs # Totalistic CA, neighborhood state
Nx = 20+1 # num of cells
TIMESTEP = 0.2
time_points = 400
Train_interval = 40



def RNN_block(x, RNN_parameters):
  # x: tf.tensor of shape [1,Nx,Gs]
  W1,b1,W2,b2,W3,b3 = RNN_parameters

  x_neighbor = 0.5*tf.concat([tf.reshape(x[:,Nx-1,:],[-1,1,Gs]), x[:,0:(Nx-1),:]], axis=1) +\
               0.5*tf.concat([x[:,1:,:], tf.reshape(x[:,0,:],[-1,1,Gs])], axis=1)

  h10 = tf.concat([x_neighbor,x],axis=2)
  a11 = tf.tensordot(h10, W1, [[2],[0]])
  h11 = tf.nn.relu(a11 + b1) #[1,Nx,32]
  a12 = tf.tensordot(h11, W2, [[2],[0]])
  h12 = tf.nn.relu(a12 + b2) #[1,Nx,32]
  a13 = tf.tensordot(h12, W3, [[2],[0]])
  f = tf.nn.sigmoid(a13 + b3)

  return f


def RNN_block_link_ko(x, RNN_parameters, LINKS):
  # x: tf.tensor of shape [1,Nx,Gs]
  # LINKS: tf.placeholder of shape [2*Gs, Gs]
  W1,b1,W2,b2,W3,b3 = RNN_parameters

  x_neighbor = 0.5*tf.concat([tf.reshape(x[:,Nx-1,:],[-1,1,Gs]), x[:,0:(Nx-1),:]], axis=1) +\
               0.5*tf.concat([x[:,1:,:], tf.reshape(x[:,0,:],[-1,1,Gs])], axis=1)

  h10 = tf.tile(tf.concat([x_neighbor,x],axis=2), [Gs,1,1]) #[Gs,Nx,2*Gs]
  h10_= h10*tf.expand_dims(tf.transpose(LINKS),1)
  a11 = tf.tensordot(h10_, W1, [[2],[0]])
  h11 = tf.nn.relu(a11 + b1) #[Gs,Nx,32]
  a12 = tf.tensordot(h11, W2, [[2],[0]])
  h12 = tf.nn.relu(a12 + b2) #[Gs,Nx,32]
  a13 = tf.tensordot(h12, W3, [[2],[0]])
  f = tf.nn.sigmoid(a13 + b3) #[Gs,Nx,Gs]
  
  f_squeeze_list = []
  for i in range(Gs):
    f_squeeze_list.append(f[i,:,i])
    f_ = tf.expand_dims(tf.stack(f_squeeze_list, axis=1),0) #[1,Nx,Gs]

  return f_


def main(_):

  if not os.path.exists('trained_RNN_models'):
    os.mkdir('trained_RNN_models')
  if not os.path.exists('trained_RNN_models/nets'):
    os.mkdir('trained_RNN_models/nets')

  ## CA ground truth
  para_all = np.genfromtxt('CA_ground_truth_models/para/paras_ground_truth.csv', dtype='float', delimiter=',')
  para = para_all[FLAGS.CAmodel_num,:]
  #b = np.reshape(para[0:(Gin+Gs)*Gs], [1,Gin+Gs,Gs])
  #K = np.reshape(para[(Gin+Gs)*Gs:2*(Gin+Gs)*Gs], [1,Gin+Gs,Gs])
  NET_ground_truth = np.reshape(para[2*(Gin+Gs)*Gs:3*(Gin+Gs)*Gs], [Gin+Gs,Gs])

  CA_init = np.zeros([BATCH_ic,Nx,Gs]); CA_init[:,Nx//2,:] = np.random.rand(BATCH_ic,Gs)

  ## RNN model
  N_width = 64
  W1 = tf.Variable(tf.truncated_normal([2*Gs, N_width], stddev=0.01))
  b1 = tf.Variable(tf.truncated_normal([N_width], stddev=0.01))
  W2 = tf.Variable(tf.truncated_normal([N_width, N_width], stddev=0.01))
  b2 = tf.Variable(tf.truncated_normal([N_width], stddev=0.01))
  W3 = tf.Variable(tf.truncated_normal([N_width, Gs], stddev=0.01))
  b3 = tf.Variable(tf.truncated_normal([Gs], stddev=0.01))
  RNN_parameters = [W1,b1,W2,b2,W3,b3]

  LINKS = tf.placeholder(tf.float32,[Gin+Gs,Gs])
  x1_init = tf.placeholder(tf.float32,[1,Nx,Gs])

  f1_list = []
  f2_list = []
  x1 = x1_init
  for t1 in range(150):
    # WT
    f_rnn1 = RNN_block(x1, RNN_parameters) #[1,Nx,Gs]
    
    # mutant f
    f_rnn2 = RNN_block_link_ko(x1, RNN_parameters, LINKS) #[1,Nx,Gs]
    if t1%2 == 0:
      f1_list.append(f_rnn1[0,:,:])
      f2_list.append(f_rnn2[0,:,:])

    # update with WT
    x1 = x1*(1-TIMESTEP) + TIMESTEP*f_rnn1

  F1 = tf.stack(f1_list, axis=0)
  F2 = tf.stack(f2_list, axis=0) #[30,Nx,Gs]
  

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet')
  
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  
  NET = np.zeros([2*Gs,Gs],'float32')
  NET_bool = np.zeros([2*Gs,Gs],'float32')

  Links_train = np.ones([2*Gs,Gs],'float32')
  for i in range(2*Gs):
    for j in range(Gs):
      if Links_train[i,j] == 0:
        NET[i,j] = 0
      else:
        Links = np.copy(Links_train)
        Links[i,j] = 0*Links[i,j]
        F1_,F2_ = sess.run([F1,F2], feed_dict={x1_init:CA_init, LINKS:Links})
        NET[i,j] = -100*np.mean(F2_[:,:,j]-F1_[:,:,j])
        NET_bool[i,j] = -(np.mean((F2_[:,:,j]-F1_[:,:,j])>=0) - np.mean((F2_[:,:,j]-F1_[:,:,j])<=0))

  print(np.round(NET,2))

  np.savetxt('trained_RNN_models/nets/model_'+repr(FLAGS.CAmodel_num)+'.csv', NET,\
             fmt='%.4g', delimiter=',')

  sess.close()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--CAmodel_num', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
