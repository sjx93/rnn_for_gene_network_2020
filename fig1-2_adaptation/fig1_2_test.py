import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig1

FLAGS = None
N_CASES = 20
Gs = 2 # num of dynamic nodes (genes)
Gin= 1 # num of input signal
TIMESTEP = 0.1 # smaller timestep for test runs
time_points = 150


def main(_):

  # folders for output
  if not os.path.exists('trajs'):
    os.mkdir('trajs')
  if not os.path.exists(repr(FLAGS.output_name)):
    os.mkdir(repr(FLAGS.output_name))
  if not os.path.exists(repr(FLAGS.output_name)+'/paras'):
    os.mkdir(repr(FLAGS.output_name)+'/paras')

  # Create session
  sess = tf.InteractiveSession()

  # case0-19, input stimulus level range from 0 to 1
  # input stimulus appears only after 40 time points
  stimuli_0 = 0.1*np.ones([N_CASES,time_points,Gin],'float32')
  stimuli_0[:,40:,0] = np.reshape(np.linspace(1/(N_CASES),1.0,N_CASES),[N_CASES,1])

  # stack all train cases
  stimuli_all = stimuli_0

  # Define model
  STIMULI = tf.placeholder(np.float32, [N_CASES,time_points,Gin])
  NN_traj, NN_paras = rnn_fig1(STIMULI, TIMESTEP) #[N_CASES,time_points,Gs]


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  # restart from saved model
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  

  # save trajectories
  # format of the output file '*_traj_g1.csv':
  # entry at (row i, column j) = g1 value at timepoint j with input strength i
  Xs_test = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all}) #[N_CASES,time_points,Gs]
  parameters = sess.run(NN_paras, feed_dict={STIMULI:stimuli_all}) #[N_CASES,time_points,Gs]
  for g in range(Gs):
    np.savetxt('trajs/run'+repr(FLAGS.output_name)+'_traj_g'+repr(g+1)+'.csv', Xs_test[:,:,g], fmt='%.4f', delimiter=',')

  # save fig
  # with 6 panels, with increasing input stimuli strength
  xs = np.linspace(0.1,time_points*TIMESTEP,time_points)
  for n in range(6):
    plt.subplot(2,3,n+1)
    plt.plot(xs, Xs_test[3*n+2,:,:],'-')
    plt.axis([0,time_points*TIMESTEP,-0.1,1])
  plt.savefig('trajs/run'+repr(FLAGS.output_name))
  plt.close()
  
  # save NN parameters
  parameter_names = ['W1','b1','W2','b2','W3','b3',' hidden_init']
  for i in range(7):
    np.savetxt(repr(FLAGS.output_name)+'/paras/'+parameter_names[i]+'.csv', parameters[i], fmt='%.4f', delimiter=',')

  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
