import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig3


FLAGS = None
N_CASES = 4
Gs = 3 # num of dynamic dims
Gin= 2 # num of input signal dims
TIMESTEP = 0.20
time_points = 120


def main(_):
  # folders for output
  if not os.path.exists('nets_fig4'):
    os.mkdir('nets_fig4')
  if not os.path.exists(repr(FLAGS.output_name+1)):
    os.mkdir(repr(FLAGS.output_name+1))

  # Create session
  sess = tf.InteractiveSession()

  # case0-3, stimuli i1
  stimuli_0 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_0[:,60:,0] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  # case4-7, stimuli i2
  stimuli_1 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_1[:,60:,1] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  # stack all test cases
  stimuli_all = np.concatenate([stimuli_0,stimuli_1],axis=0) #[N_CASES,time_points,Gin]


  # Define model
  STIMULI = tf.placeholder(np.float32, [N_CASES,time_points,Gin])
  LINKS = tf.placeholder(np.float32, [Gin+Gs,Gs])
  NN_traj = rnn_fig3(STIMULI, LINKS, TIMESTEP, Gs) #[N_CASES,time_points,Gs]


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  
  
  # links used during training
  links_train = np.genfromtxt(repr(FLAGS.output_name)+'/links.csv', delimiter=',', dtype='float32') #[Gin+Gs,Gs]

  # "wild type" trajectory for reference
  Xs_wt = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:links_train}) #[N_CASES,time_points,Gs] 

  NET = np.zeros([Gin+Gs, Gs],'float32')
  ERR = 10000*np.ones([Gin+Gs, Gs],'float32')
  for i in range(Gin+Gs):
    for j in range(Gs):
      if (links_train[i,j] != 0):
        links = np.copy(links_train)
        links[i,j] = FLAGS.lambda_factor
        Xs_KO = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:links*links_train}) #[N_CASES,time_points,Gs]
        NET[i,j] = np.mean(Xs_wt[:,:,j] - Xs_KO[:,:,j])
          
        # total squared error introduced by this knockout
        if (FLAGS.keep_input_links==1):
          if ((i+1)>Gin):
            ERR[i,j] = np.mean((Xs_wt[:,:,j] - Xs_KO[:,:,j])**2)
        else:
          ERR[i,j] = np.mean((Xs_wt[:,:,j] - Xs_KO[:,:,j])**2)

  np.savetxt('nets_fig4/run'+repr(FLAGS.output_name)+'.csv',NET, fmt='%.4f', delimiter=',')
  print('current regulation network (strength)')
  print(NET)
  print('current regulation network (sign)')
  print(np.sign(NET))

  # write new links
  new_links = links_train*(ERR!=np.min(ERR))
  np.savetxt(repr(FLAGS.output_name+1) +'/links.csv',new_links, fmt='%.4f', delimiter=',')
  print('overall change induced by link knockout:')
  print(np.sqrt(ERR))
  print('new links:')
  print(new_links)

  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=61, help='***')
  parser.add_argument('--lambda_factor', type=float, default=0.5, help='***')
  parser.add_argument('--keep_input_links', type=int, default=1, help='0:off, 1:on')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
