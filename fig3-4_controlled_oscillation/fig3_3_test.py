import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig3

FLAGS = None
N_CASES = 40
Gs = 2
Gin= 2
TIMESTEP = 0.10
time_points = 300
  

def main(_):

  # folders for output
  if not os.path.exists('trajs'):
    os.mkdir('trajs')

  # Create session
  sess = tf.InteractiveSession()

  # case0-14, stimuli i1
  stimuli_0 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_0[:,60:,0] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  # case15-29, stimuli i2
  stimuli_1 = np.zeros([N_CASES//2,time_points,Gin],'float32')
  stimuli_1[:,60:,1] = np.reshape(np.linspace(1/(N_CASES//2),1.0,N_CASES//2),[N_CASES//2,1])

  # stack all cases
  stimuli_all = np.concatenate([stimuli_0,stimuli_1],axis=0) #[N_CASES,time_points,Gin]

  # NN model
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
  

  # save trajectories
  # output format:
  # traj_g1.shape=[40,timepoints]
  # traj_g1[0:20,:]: 20 trajectories with I1 = 0.05 to 1.0 and I2=0
  # traj_g1[20:,:]: 20 trajectories with I2 = 0.05 to 1.0 and I1=0
  Xs_test = sess.run(NN_traj, feed_dict={LINKS:np.ones([Gin+Gs,Gs],'float32'), STIMULI:stimuli_all})
  for g in range(Gs):
    np.savetxt('trajs/run'+repr(FLAGS.output_name)+'_traj_g'+repr(g+1)+'.csv', Xs_test[:,:,g], fmt='%.4f', delimiter=',')

  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=3, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
