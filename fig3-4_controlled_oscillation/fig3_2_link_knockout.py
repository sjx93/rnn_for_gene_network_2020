import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig3

FLAGS = None
N_CASES = 8
Gs = 2
Gin= 2
TIMESTEP = 0.10
time_points = 300


def main(_):

  # folders for output
  if not os.path.exists('nets_fig3d-f'):
    os.mkdir('nets_fig3d-f')

  # Create session
  sess = tf.InteractiveSession()

  # case0-3, stimuli i1
  stimuli_0 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_0[:,60:,0] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  # case4-7, stimuli i2
  stimuli_1 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_1[:,60:,1] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  stimuli_all = np.concatenate([stimuli_0,stimuli_1],axis=0) #[N_CASES,time_points,Gin]

  STIMULI = tf.placeholder(np.float32, [N_CASES,time_points,Gin])
  LINKS = tf.placeholder(np.float32, [Gin+Gs,Gs])
  NN_traj = rnn_fig3(STIMULI, LINKS, TIMESTEP) #[N_CASES,time_points,Gs]


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  
  # "wild type" trajectory, with all links allowed
  Xs_wt = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:np.ones([Gin+Gs,Gs],'float32')}) #[N_CASES,time_points,Gs] 

  NET = np.zeros([Gin+Gs, Gs],'float32')
  ERR = np.zeros([Gin+Gs, Gs],'float32')
  for i in range(Gin+Gs):
    for j in range(Gs):
      # gj level change
      links = np.ones([Gin+Gs,Gs],'float32')
      links[i,j] = FLAGS.lambda_factor
      Xs_KO = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:links}) #[N_CASES,time_points,Gs]
      NET[i,j] = np.mean(Xs_wt[:,:,j] - Xs_KO[:,:,j])

      # total squared error introduced by this knockout
      links = np.ones([Gin+Gs,Gs],'float32')
      links[i,j] = 0.0
      Xs_KO = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:links}) #[N_CASES,time_points,Gs]
      ERR[i,j] = np.sqrt(np.mean((Xs_wt[:,:,j] - Xs_KO[:,:,j])**2))

      
  np.savetxt('nets_fig3d-f/run'+repr(FLAGS.output_name)+'.csv',NET, fmt='%.4f', delimiter=',')
  np.savetxt('nets_fig3d-f/err'+repr(FLAGS.output_name)+'.csv',NET, fmt='%.4f', delimiter=',')
  print('current regulation network (strength)')
  print(NET)
  print('current regulation network (sign)')
  print(np.sign(NET))
  print('overall change induced by link knockout:')
  print(ERR)
  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  parser.add_argument('--lambda_factor', type=float, default=0.9, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
