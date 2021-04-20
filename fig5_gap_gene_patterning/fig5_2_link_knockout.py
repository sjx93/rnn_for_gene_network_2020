import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


Gs = 4
N1 = 64
FLAGS = None


def main(_):

  if not os.path.exists('nets'):
    os.mkdir('nets')

  # Create the model
  sess = tf.InteractiveSession()

  xs = np.linspace(0.0,1.0,N1)
  bcd1_init = np.reshape(np.float32( np.exp(-xs/0.2) ),[1,N1,1])
  tsl1_init = np.reshape(np.float32( np.exp(-xs/0.05) + np.exp(-(1-xs)/0.05) ),[1,N1,1])

  Links_train = np.ones([6,4],'float32')
  LINKS = tf.placeholder(tf.float32,[6,4])  

  W1 = tf.Variable(tf.truncated_normal([Gs+2, 32], stddev=0.1))
  b1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
  W2 = tf.Variable(tf.truncated_normal([32, 32], stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
  W3 = tf.Variable(tf.truncated_normal([32,Gs], stddev=0.1))
  b3 = tf.Variable(tf.truncated_normal([Gs], stddev=0.1))

  TIME_STEP = 0.1
  DECAYp = 1.0

  # wt
  bcd1 = bcd1_init
  tsl1 = tsl1_init

  x1 = 0.0*tf.ones([1,N1,Gs]) #[1,N1,Gs]

  for t11 in range(30):
    x1p = tf.tile(x1,[Gs,1,1])
    h10 = tf.concat([x1p,tf.tile(bcd1,[Gs,1,1]),tf.tile(tsl1,[Gs,1,1])], axis=2) #[Gs,N1,Gs+2]
    h10_= h10*tf.expand_dims(tf.transpose(LINKS),1)
    a11 = tf.tensordot(h10_, W1, [[2],[0]])
    h11 = tf.nn.relu(a11 + b1) #[Gs,N1,32]
    a12 = tf.tensordot(h11, W2, [[2],[0]])
    h12 = tf.nn.relu(a12 + b2) #[Gs,N1,32]
    a13 = tf.tensordot(h12, W3, [[2],[0]])
    h13 = tf.nn.sigmoid(a13 + b3) #[Gs,N1,Gs]
    h13_= tf.expand_dims(tf.stack([h13[0,:,0],h13[1,:,1],h13[2,:,2],h13[3,:,3]], axis=1),0) #[1,N1,Gs]
    x1 = x1*(1-TIME_STEP*DECAYp) + TIME_STEP*h13_

  y1 = x1[0,:,:]


  tf.global_variables_initializer().run()
  # saving and loading networks
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(repr(FLAGS.output_name)+'/savenet')
  
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old network weights")
  

  NET = np.zeros([6,4],'float32')
  ERR = np.zeros([6,4],'float32')
  
  X_wt = np.concatenate([sess.run(y1, feed_dict={LINKS:Links_train}),bcd1[0,:,:],tsl1[0,:,:]], axis=1) #[N1,Gs+2]
  
  for i in range(6):
    for j in range(4):
      if Links_train[i,j] == 0:
        NET[i,j] = 0
        ERR[i,j] = 100
      else:
        Links = np.copy(Links_train)
        Links[i,j] = 0.0*Links[i,j]
        X_mut = sess.run(y1, feed_dict={LINKS:Links})
        NET[i,j] = -np.sum(X_mut[:,j]-X_wt[:,j])
        ERR[i,j] = np.mean((X_mut-X_wt[:,0:Gs])**2)

  print('current regulation network (strength)')
  print(NET)
  np.savetxt('nets/run_'+repr(FLAGS.output_name)+'.csv', NET, fmt='%.4g', delimiter=',')
  print('current regulation network (sign)')
  print(np.sign(NET))
  print('overall change induced by link knockout:')
  print(ERR)
  np.savetxt('nets/err_'+repr(FLAGS.output_name)+'.csv', ERR, fmt='%.4g', delimiter=',')


  sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
