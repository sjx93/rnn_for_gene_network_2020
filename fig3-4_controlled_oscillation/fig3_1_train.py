import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig3
from target_response_curves import spikes

FLAGS = None
N_CASES = 3
Gs = 2 # num of dynamic nodes (genes)
Gin= 2 # num of input signal
TIMESTEP = 0.2
time_points = 60


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
  LINKS = tf.placeholder(np.float32, [Gin+Gs,Gs])
  TARGET = tf.placeholder(np.float32, [N_CASES,time_points])
  NN_traj = rnn_fig3(STIMULI, LINKS, TIMESTEP, Gs) #[N_CASES,time_points,Gs]
  
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
  stimuli_0 = np.zeros([time_points,Gin],'float32')
  target_0 = 0.1*np.ones([time_points],'float32')

  # Case-1, ocillation under Input-1
  stimuli_2 = np.zeros([time_points,Gin],'float32')
  stimuli_2[:,0] = 0.8
  position = np.array([0,15,30,45])
  height = np.array([1,1,1,1])
  duration = np.array([10,10,10,10]) 
  target_2 = spikes(time_points, position,height,duration)
  
  # Case-2, steady-state responce under Input-2
  stimuli_4 = np.zeros([time_points,Gin],'float32')
  stimuli_4[:,1] = 0.8
  target_4 = 0.8*np.ones([time_points],'float32')
  
  # stack all train cases
  stimuli_all = np.stack([stimuli_0,stimuli_2,stimuli_4],axis=0) #[BATCH,time_points,Gin]
  target_all  = np.stack([target_0,target_2,target_4],axis=0) #[BATCH,time_points]
  
  for i in range(8001):
  
    # apply gradient
    sess.run(train_step,\
             feed_dict={LINKS:np.ones([Gin+Gs,Gs],'float32'), STIMULI:stimuli_all, TARGET:target_all})

    # Test
    if i%100 == 0:
      monitor = sess.run([loss0_1, loss1, loss2],\
                feed_dict={LINKS:np.ones([Gin+Gs,Gs],'float32'), STIMULI:stimuli_all, TARGET:target_all})
      print('step%g, loss0:%.4g, loss1:%.4g, loss2:%.4g,'%(i, monitor[0], monitor[1], monitor[2]))
      loss_writer.writerow([i, monitor[0], monitor[1], monitor[2]])

      if i%500 == 0:
        xs = np.linspace(0.0,(time_points-1), time_points)
        Xs_test = sess.run(NN_traj,\
                  feed_dict={LINKS:np.ones([Gin+Gs,Gs],'float32'), STIMULI:stimuli_all}) #[N_CASES,time_points,Gs] 

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
