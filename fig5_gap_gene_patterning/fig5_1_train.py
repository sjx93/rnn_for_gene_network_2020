import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


BATCH = 1
NUM_GENES = 4
N1 = 91
EXPLICIT_NOISE_LEVEL = 0.05
FLAGS = None


def main(_):

  if not os.path.exists(repr(FLAGS.output_name)):
    os.mkdir(repr(FLAGS.output_name))
  if not os.path.exists(repr(FLAGS.output_name)+'/savenet'):
    os.mkdir(repr(FLAGS.output_name)+'/savenet')
  if not os.path.exists(repr(FLAGS.output_name)+'/train'):
    os.mkdir(repr(FLAGS.output_name)+'/train')

  # Create the model
  sess = tf.InteractiveSession()

  xs = np.linspace(0.05,0.95,N1)
  bcd1_init = np.reshape(np.float32( np.exp(-xs/0.2) ),[1,N1,1])
  tsl1_init = np.reshape(np.float32( np.exp(-xs/0.05) + np.exp(-(1-xs)/0.05) ),[1,N1,1])

  wt_ = np.genfromtxt('Data_frame/FlyEX_nc14T7_gap-genes.csv', delimiter=',', dtype='float32') #[N1,5]

  W1 = tf.Variable(tf.truncated_normal([NUM_GENES+2, 32], stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([32,NUM_GENES], stddev=0.1))
  b3 = tf.Variable(tf.truncated_normal([NUM_GENES], stddev=0.1))

  TIME_STEP = 0.1
  DECAYp = 1.0

  # wt
  bcd1 = bcd1_init
  tsl1 = tsl1_init

  x1 = 0.0*tf.ones([1,N1,NUM_GENES]) #[1,N1,NUM_GENES]

  for t11 in range(30):
    x1p = x1
    h10 = tf.concat([x1p,bcd1,tsl1], axis=2)
    a11 = tf.tensordot(h10, W1, [[2],[0]])
    h11 = tf.nn.relu(a11 + b1) #[1,N1,32]
    a12 = tf.tensordot(h11, W2, [[2],[0]])
    h12 = tf.nn.relu(a12 + b2) #[1,N1,32]
    a13 = tf.tensordot(h12, W3, [[2],[0]])
    h13 = tf.nn.sigmoid(a13 + b3) + tf.truncated_normal([N1,NUM_GENES], stddev=EXPLICIT_NOISE_LEVEL) #[1,N1,NUM_GENES]
    x1 = x1*(1-TIME_STEP*DECAYp) + TIME_STEP*h13

  y1 = x1[0,:,:]
  dist1 = tf.reduce_mean((y1-wt_[:,0:NUM_GENES])**2)


  # Regularization
  L2 = tf.reduce_sum(W1**2) + tf.reduce_sum(W2**2) + tf.reduce_sum(W3**2)


  # Define loss and optimizer
  train_step = tf.train.AdamOptimizer(0.001).minimize(tf.sqrt(dist1))

  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  '''
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  '''

  # Train
  accuracy_writer = csv.writer(open(repr(FLAGS.output_name)+'/savenet/accuracy.csv', 'w'))

  for i in range(4001):
    # apply grad
    sess.run(train_step)

    # Test
    if i%100 == 0:
      accuracy = sess.run([dist1, dist1, dist1])
      print('step%g, err:%.4g, %.4g, %.4g'%(i, accuracy[0], accuracy[1], accuracy[2]))
      accuracy_writer.writerow([i, accuracy[0], accuracy[1], accuracy[2]])

    
      if i%1000 == 0:
        X_test = sess.run(tf.concat([x1,bcd1,tsl1],axis=2))

        plt.plot(np.linspace(0.05,0.95,N1), np.reshape(X_test,[N1,NUM_GENES+2]))
        plt.axis([0,1,0,1.2])

        plt.savefig(repr(FLAGS.output_name)+'/train/round%d'%i)
        plt.close()
    
    if i%4000 == 3999:
      saver.save(sess, repr(FLAGS.output_name)+'/savenet/gap-network' , global_step = i)

  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
