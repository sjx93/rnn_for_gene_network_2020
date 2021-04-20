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
FLAGS = None
mut_name = ['wt','bcd-','tsl-',\
            'bcd1X','bcd4X',\
            'hb-','kr-','kni-','gt-']

def main(_):

  if not os.path.exists(repr(FLAGS.run_num)+'/test'):
    os.mkdir(repr(FLAGS.run_num)+'/test')
  if not os.path.exists(repr(FLAGS.run_num)+'/test/wt'):
    os.mkdir(repr(FLAGS.run_num)+'/test/wt')

  # Create the model
  sess = tf.InteractiveSession()

  xs = np.linspace(0.05,0.95,N1)
  bcd1_init = np.reshape(np.float32( np.exp(-xs/0.2) ),[1,N1,1])
  tsl1_init = np.reshape(np.float32( np.exp(-xs/0.05) + np.exp(-(1-xs)/0.05) ),[1,N1,1])

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
  X1_tf =[]
  for t11 in range(10):
    for t12 in range(3):
      x1p = x1
      h10 = tf.concat([x1p,bcd1,tsl1], axis=2)
      a11 = tf.tensordot(h10, W1, [[2],[0]])
      h11 = tf.nn.relu(a11 + b1) #[1,N1,32]
      a12 = tf.tensordot(h11, W2, [[2],[0]])
      h12 = tf.nn.relu(a12 + b2) #[1,N1,32]
      a13 = tf.tensordot(h12, W3, [[2],[0]])
      h13 = tf.nn.sigmoid(a13 + b3)
      x1 = x1*(1-TIME_STEP*DECAYp) + TIME_STEP*h13

    X1_tf.append(x1[0,:,:]) #[N1,4]


  # mutants
  Case   = np.array([[1,1,1, 1,1,1,1],[0,1,1, 1,1,1,1],[1,1,0, 1,1,1,1],\
                     [0.5,1,1, 1,1,1,1],[2,1,1, 1,1,1,1],\
                     [1,1,1, 0,1,1,1],[1,1,1, 1,0,1,1],[1,1,1, 1,1,0,1],[1,1,1, 1,1,1,0],\
                    ],'float32')

  bcd3 = tf.tile(bcd1_init, [len(mut_name),1,1])*tf.reshape(Case[:,0], [len(mut_name),1,1])
  tsl3 = tf.tile(tsl1_init, [len(mut_name),1,1])*tf.reshape(Case[:,2], [len(mut_name),1,1])
  x3 = 0.0*tf.ones([len(mut_name),N1,NUM_GENES]) #[len(mut_name),N1,NUM_GENES]

  for t31 in range(30):
    x3p = x3
    h30 = tf.concat([x3p,bcd3,tsl3], axis=2)
    a31 = tf.tensordot(h30, W1, [[2],[0]])
    h31 = tf.nn.relu(a31 + b1) #[1,N1,32]
    a32 = tf.tensordot(h31, W2, [[2],[0]])
    h32 = tf.nn.relu(a32 + b2) #[1,N1,32]
    a33 = tf.tensordot(h32, W3, [[2],[0]])
    h33 = tf.nn.sigmoid(a33 + b3)
    x3 = x3*(1-TIME_STEP*DECAYp) + TIME_STEP*h33*tf.expand_dims(Case[:,3:],1)

  X3_tf = x3[:,:,:] #[7,N1,4]


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.run_num)+'/savenet')
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")


  X1_test, X3_test = sess.run([X1_tf, X3_tf])

  for i in range(len(X1_test)):
    plt.plot(np.linspace(0,1,N1), X1_test[i])
    plt.axis([0,1,0,1.2])
    plt.savefig(repr(FLAGS.run_num)+'/test/wt/t='+repr(i))
    plt.close()
    np.savetxt(repr(FLAGS.run_num)+'/test/wt/t='+repr(i)+'.csv', X1_test[i], fmt='%.4g', delimiter=',')


  for i in range(len(mut_name)):
    x_mut_plot = np.reshape(X3_test[i,:,:], [N1,4])
    plt.plot(np.linspace(0,1,N1), x_mut_plot)
    plt.axis([0,1,0,1.2])
    plt.savefig(repr(FLAGS.run_num)+'/test/%d_'%(i+1)+mut_name[i])
    plt.close()
    np.savetxt(repr(FLAGS.run_num)+'/test/%d_'%(i+1)+mut_name[i]+'.csv', x_mut_plot, fmt='%.4g', delimiter=',')
      
  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_num', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
