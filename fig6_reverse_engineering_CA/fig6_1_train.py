import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FLAGS = None
BATCH_ic = 16+1 # different initial conditions
BATCH_rnn= 8
Gs = 10 # num of dynamic dims
Gin= Gs # Totalistic CA, neighborhood state
Nx = 20+1 # num of cells
TIMESTEP = 0.2
time_points = 8*40
Train_interval = 40

def Hill_model(x,b,K,links):
  # x, current input and state [BATCH_ic,Nx,Gin+Gs]
  # b, max_rate for Hill activation term [BATCH_ic,Gin+Gs,Gs]
  # K, Micheales constant [BATCH_ic,Gin+Gs, Gs]
  # links, network topology +1/0/-1, [Gin+Gs, Gs]

  Hill_n = 2
  f0 = (np.reshape(x,[BATCH_ic,Nx,Gin+Gs,1])**Hill_n)/ \
       (np.tile(np.reshape(K,[1,1,Gin+Gs,Gs]),[BATCH_ic,1,1,1])**Hill_n + np.reshape(x,[BATCH_ic,Nx,Gin+Gs,1])**Hill_n) #[BATCH_ic,Nx,Gin+Gs,Gs]

  f_activation1 = f0*np.tile(np.reshape(b,[1,1,Gin+Gs,Gs]),[BATCH_ic,1,1,1])*np.reshape(links==1,[1,1,Gin+Gs,Gs])
  f_activation2 = np.sum(f_activation1,axis=2) #[BATCH_ic,Nx,Gs]

  f_inhibition1 = (1-f0)*np.reshape(links==-1,[1,1,Gin+Gs,Gs]) + np.reshape(links!=-1,[1,1,Gin+Gs,Gs])
  f_inhibition2 = np.prod(f_inhibition1,axis=2) #[BATCH_ic,Nx,Gs]

  f = (f_activation2)*f_inhibition2 #[BATCH_ic,Nx,Gs]
  return f


def Hill_model_dynamics_test(x_init, b,K,links):
  # x_init.shape = [BATCH_ic,Nx,Gs]
  gamma = 1
  X0 = np.zeros([BATCH_ic,time_points,Nx,Gs])
  x0 = x_init

  for t0 in range(time_points):

    x0_neighbor = 0.5*np.concatenate([np.reshape(x0[:,Nx-1,:],[BATCH_ic,1,Gs]), x0[:,0:(Nx-1),:]], axis=1) +\
                  0.5*np.concatenate([x0[:,1:,:], np.reshape(x0[:,0,:],[BATCH_ic,1,Gs])], axis=1)

    f0 = Hill_model(np.concatenate([x0_neighbor,x0],axis=2),b,K,links)
    x0 = (1-gamma*TIMESTEP)*x0 + f0*TIMESTEP #[BATCH_ic,Nx,Gs]
    X0[:,t0,:,:] = x0

  return X0 #[:,range(0,timepoints,10),:,:]
  

def RNN_block(x, RNN_parameters):
  # x: tf.tensor of shape [~,Nx,Gs]
  W1,b1,W2,b2,W3,b3 = RNN_parameters

  x_neighbor = 0.5*tf.concat([tf.reshape(x[:,Nx-1,:],[-1,1,Gs]), x[:,0:(Nx-1),:]], axis=1) +\
               0.5*tf.concat([x[:,1:,:], tf.reshape(x[:,0,:],[-1,1,Gs])], axis=1)

  h10 = tf.concat([x_neighbor,x],axis=2)
  a11 = tf.tensordot(h10, W1, [[2],[0]])
  h11 = tf.nn.relu(a11 + b1) #[1,N1,32]
  a12 = tf.tensordot(h11, W2, [[2],[0]])
  h12 = tf.nn.relu(a12 + b2) #[1,N1,32]
  a13 = tf.tensordot(h12, W3, [[2],[0]])
  f = tf.nn.sigmoid(a13 + b3)

  return f


def main(_):

  if not os.path.exists('trained_RNN_models'):
    os.mkdir('trained_RNN_models')
  if not os.path.exists('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)):
    os.mkdir('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num))
  if not os.path.exists('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet'):
    os.mkdir('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet')
  if not os.path.exists('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/train'):
    os.mkdir('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/train')

  ## CA fitting target
  para_all = np.genfromtxt('CA_ground_truth_models/para/paras_ground_truth.csv', dtype='float', delimiter=',')
  para = para_all[FLAGS.CAmodel_num,:]
  b = np.reshape(para[0:(Gin+Gs)*Gs], [1,Gin+Gs,Gs])
  K = np.reshape(para[(Gin+Gs)*Gs:2*(Gin+Gs)*Gs], [1,Gin+Gs,Gs])
  links = np.reshape(para[2*(Gin+Gs)*Gs:3*(Gin+Gs)*Gs], [Gin+Gs,Gs])

  x_init = np.zeros([BATCH_ic,Nx,Gs]); x_init[:,Nx//2,:] = np.random.rand(BATCH_ic,Gs)
  X_CA0 = np.float32(Hill_model_dynamics_test(x_init,b,K,links)) #[BATCH_ic,time_points,Nx,Gs]
  X_CA = X_CA0/np.max(X_CA0,axis=(0,1,2)) #normalization
  np.savetxt('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet/norm_factor.csv',\
             np.max(X_CA0,axis=(0,1,2)), fmt='%.8g', delimiter=',')

  # prepare trainng set
  X_CA_train_list = []
  for i_initic in range(BATCH_ic-1):
    for i_tps in range((time_points-Train_interval)//10):
      t_start = 10*i_tps
      t_end = 10*i_tps+Train_interval
      X_CA_train_list.append(X_CA[i_initic,range(t_start,t_end,10),:,:])
  X_CA_train0 = np.stack(X_CA_train_list,axis=0) #[28*8,4,Nx,Gs]
  n_perm = np.arange((BATCH_ic-1)*((time_points-Train_interval)//10))
  np.random.shuffle(n_perm)
  X_CA_train = np.copy(X_CA_train0[n_perm,:,:,:]) #[28*8,4,Nx,Gs]

  # prepare validation set
  X_CA_validation_list = []
  for i_tps in range((time_points-Train_interval)//10):
    t_start = 10*i_tps
    t_end = 10*i_tps+Train_interval
    X_CA_validation_list.append(X_CA[i_initic,range(t_start,t_end,10),:,:])
  X_CA_validation = np.stack(X_CA_validation_list,axis=0) #[28,4,Nx,Gs]
  

  ## RNN model
  N_width = 64
  W1 = tf.Variable(tf.truncated_normal([2*Gs, N_width], stddev=0.001))
  b1 = tf.Variable(tf.truncated_normal([N_width], stddev=0.001))
  W2 = tf.Variable(tf.truncated_normal([N_width, N_width], stddev=0.001))
  b2 = tf.Variable(tf.truncated_normal([N_width], stddev=0.001))
  W3 = tf.Variable(tf.truncated_normal([N_width, Gs], stddev=0.001))
  b3 = tf.Variable(tf.truncated_normal([Gs], stddev=0.001))
  RNN_parameters = [W1,b1,W2,b2,W3,b3]

  x1_target= tf.placeholder(tf.float32,[None,Train_interval//10,Nx,Gs])
  x1 = x1_target[:,0,:,:]
  loss = 0
  #X1_tf = []
  for t1 in range(Train_interval-1):
    f_rnn = RNN_block(x1, RNN_parameters)
    x1 = x1*(1-TIMESTEP) + TIMESTEP*f_rnn
    if t1%10 == (10-1):
      loss += tf.reduce_mean((x1-x1_target[:,(t1+1)//10,:,:])**2)

  train_step = tf.train.AdamOptimizer(0.01).minimize(tf.sqrt(loss))
  
  
  # RNN model for test
  X2_tf = []
  x2_init = tf.placeholder(tf.float32,[1,Nx,Gs])
  x2 = x2_init
  for t2 in range(150):
    f_rnn2 = RNN_block(x2, RNN_parameters)
    x2 = x2*(1-TIMESTEP) + TIMESTEP*f_rnn2
    if t2%5 == 0:
      X2_tf.append(x2[0,:,0:3])
  

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet')
  '''
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
  '''

  # Train
  accuracy_writer = csv.writer(open('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet/accuracy.csv', 'w'))
    
  for i in range(12001):
    # choose minibatch
    n_train = np.random.choice(np.arange(X_CA_train.shape[0]),BATCH_rnn,replace=False)
    train_data = X_CA_train[n_train,:,:,:] # [BATCH_rnn,Train_interval/10,Nx,Gs]

    # apply grad
    sess.run(train_step, feed_dict={x1_target: train_data})

    if i%100 == 0:
      train_loss = sess.run(loss, feed_dict={x1_target: train_data})
      validation_loss = sess.run(loss, feed_dict={x1_target: X_CA_validation})
      print('step%g, train_loss:%.4g, test_loss:%.4g'%(i, train_loss, validation_loss))
      accuracy_writer.writerow([i, train_loss, validation_loss])
      
      if i%4000 == 0:
        X2_test = sess.run(tf.stack(X2_tf,axis=0), feed_dict={x2_init: np.expand_dims(X_CA[BATCH_ic-1,0,:,:],0)})
        plt.subplot(1,2,1)
        plt.imshow(X2_test)
        plt.subplot(1,2,2)
        plt.imshow(X_CA[BATCH_ic-1,range(0,150,5),:,0:3])
        plt.savefig('trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/train/step'+repr(i))
      

    if i%4000 == 3999:
      saver.save(sess, 'trained_RNN_models/model_'+repr(FLAGS.CAmodel_num)+'/savenet/CA-network' , global_step = i)

  sess.close()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--CAmodel_num', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
