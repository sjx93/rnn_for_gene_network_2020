import argparse
import sys
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from rnn_models import rnn_fig2

FLAGS = None
N_CASES = 4
Gs = 2 # num of dynamic nodes (genes)
Gin= 1 # num of input signal
TIMESTEP = 0.2
time_points = 60


def main(_):
  # folders for output
  if not os.path.exists('nets'):
    os.mkdir('nets')
  if not os.path.exists(repr(FLAGS.output_name+1)):
    os.mkdir(repr(FLAGS.output_name+1))

  # Create session
  sess = tf.InteractiveSession()

  # case0-3, evaluate link-mutation effect with multiple input strengths
  stimuli_0 = 0.1*np.ones([N_CASES,time_points,Gin],'float32')
  stimuli_0[:,20:,0] = np.reshape(np.linspace(1/(N_CASES),1.0,N_CASES),[N_CASES,1])

  # stack all test cases
  stimuli_all = stimuli_0 #[N_CASES,time_points,Gin]

  # Define model
  STIMULI = tf.placeholder(np.float32, [N_CASES,time_points,Gin])
  LINKS = tf.placeholder(tf.float32, [Gin+Gs,Gs])
  NN_traj = rnn_fig2(STIMULI, LINKS, TIMESTEP) #output shape:[N_CASES,time_points,Gs]

  tf.global_variables_initializer().run()
  # loading trained model
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

  # knockdown each links and run the model
  NET = np.zeros([Gin+Gs, Gs],'float32')
  ERR = 10000*np.ones([Gin+Gs, Gs],'float32')
  for i in range(Gin+Gs):
    for j in range(Gs):
      if (links_train[i,j] != 0):
        links = np.copy(links_train)
        links[i,j] = FLAGS.lambda_factor
        Xs_KO = sess.run(NN_traj, feed_dict={STIMULI:stimuli_all, LINKS:links*links_train}) #[N_CASES,time_points,Gs]
        NET[i,j] = np.mean(Xs_wt[:,:,j] - Xs_KO[:,:,j]) # averaged change in gj level
        
        # total squared error introduced by this knockout
        if (FLAGS.keep_input_links==1):
          if ((i+1)>Gin):
            ERR[i,j] = np.mean((Xs_wt[:,:,j] - Xs_KO[:,:,j])**2)
        else:
          ERR[i,j] = np.mean((Xs_wt[:,:,j] - Xs_KO[:,:,j])**2)

  np.savetxt('nets/run'+repr(FLAGS.output_name)+'.csv',NET, fmt='%.4f', delimiter=',')
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
  parser.add_argument('--output_name', type=int, default=3, help='***')
  parser.add_argument('--lambda_factor', type=float, default=0.0, help='***')
  parser.add_argument('--keep_input_links', type=int, default=0, help='0:off, 1:on')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
