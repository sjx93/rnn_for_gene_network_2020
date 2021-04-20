import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig1
from target_response_curves import adapt_pulse_double_exp as adapt_pulse

FLAGS = None
N_CASES = 3
Gs = 2 # num of dynamic nodes (genes)
Gin= 1 # num of input signal
TIMESTEP = 0.2
time_points = 40


def main(_):

  # folders for output
  if not os.path.exists(repr(FLAGS.output_name)):
    os.mkdir(repr(FLAGS.output_name))
  if not os.path.exists(repr(FLAGS.output_name)+'/train'):
    os.mkdir(repr(FLAGS.output_name)+'/train')
  if not os.path.exists(repr(FLAGS.output_name)+'/savenet'):
    os.mkdir(repr(FLAGS.output_name)+'/savenet')

  # Create session
  sess = tf.InteractiveSession()

  # Define model and Loss
  STIMULI = tf.placeholder(np.float32, [N_CASES,time_points,Gin])
  TARGET = tf.placeholder(np.float32, [N_CASES,time_points])
  NN_traj, _ = rnn_fig1(STIMULI, TIMESTEP) #[N_CASES,time_points,Gs]
  
  loss0_0 = tf.reduce_sum((NN_traj[0,:,:]-NN_traj[0,0,:])**2)
  loss0_1 = tf.reduce_sum((NN_traj[0,:,0]-TARGET[0,:])**2) #Case-0
  loss1 = tf.reduce_sum((NN_traj[1,:,0]-TARGET[1,:])**2) #Case-1
  loss2 = tf.reduce_sum((NN_traj[2,:,0]-TARGET[2,:])**2) #Case-2
  loss  = loss0_0 + loss0_1 + loss1 + loss2 # all

  # Define optimizer
  train_step = tf.train.RMSPropOptimizer(0.001).minimize(tf.sqrt(loss))


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  '''
  # restart from saved model if necessary
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
  '''

  # Train
  loss_writer = csv.writer(open(repr(FLAGS.output_name)+'/loss.csv', 'w'))
  
  # Case-0, fix-point without stimuli
  stimuli_0 = 0.1*np.ones([time_points,Gin],'float32')
  target_0 = 0.4*np.ones([time_points],'float32')

  # Case-1, high peak responce with high stimuli
  stimuli_1 = np.zeros([time_points,Gin],'float32')
  stimuli_1[:,0] = 1.0
  target_1 = adapt_pulse(time_points, height=1.0)+0.4
  
  for i in range(2001):
    # Case-2, random stimuli strength
    stimuli_level = np.random.rand()
    stimuli_2 = 0.1*np.ones([time_points,Gin],'float32')
    stimuli_2[:,0] = stimuli_level
    target_2 = adapt_pulse(time_points, height=stimuli_level)+0.4
    
    # stack all train cases
    stimuli_all = np.stack([stimuli_0,stimuli_1,stimuli_2],axis=0) #[N_CASES,time_points,Gin]
    target_all  = np.stack([target_0,target_1,target_2],axis=0) #[N_CASES,time_points]
    
    # apply gradient
    sess.run(train_step, feed_dict={STIMULI:stimuli_all, TARGET:target_all})

    # Test
    if i%100 == 0:
      monitor = sess.run([loss0_1, loss1, loss2], feed_dict={STIMULI:stimuli_all, TARGET:target_all})
      print('step%g, loss0:%.4g, loss1:%.4g, loss2:%.4g,'%(i, monitor[0], monitor[1], monitor[2]))
      loss_writer.writerow([i, monitor[0], monitor[1], monitor[2]])

      if i%500 == 0:
        xs = np.linspace(0.0,(time_points-1), time_points)
        Xs_test = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all}) #[N_CASES,time_points,Gs] 

        for n in range(N_CASES):
          plt.subplot(3,2,n+1)
          plt.plot(xs, Xs_test[n,:,:],'-', xs, target_all[n,:], ':k')
          plt.axis([0,time_points-1,-0.1,1])

        plt.savefig(repr(FLAGS.output_name)+'/train/step'+repr(i))
        plt.close()

    if i%2000 == 1999:
      saver.save(sess, repr(FLAGS.output_name)+'/savenet/dyn-network' , global_step = i)

  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
