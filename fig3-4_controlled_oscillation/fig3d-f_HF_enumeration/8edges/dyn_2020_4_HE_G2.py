import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FLAGS = None
BATCH = 8000
Gs = 2 # num of dynamic nodes (genes)
Gin= 2 # num of input signal
TIMESTEP = 0.25
time_points = 200

def Hill_model(x,b,K,links):
  # x, current input and state [BATCH,Gin+Gs]
  # b, max_rate for Hill activation term [BATCH,Gin+Gs,Gs]
  # K, Micheales constant [BATCH,Gin+Gs, Gs]
  # links, network topology +1/0/-1, [Gin+Gs, Gs]

  Hill_n = 2
  f0 = ((np.reshape(x,[BATCH,Gin+Gs,1]))**Hill_n)/(K**Hill_n+(np.reshape(x,[BATCH,Gin+Gs,1]))**Hill_n) #[BATCH,Gin+Gs,Gs]

  f_activation1 = f0*b*np.reshape(links==1,[1,Gin+Gs,Gs])
  f_activation2 = np.sum(f_activation1,axis=1) #[BATCH,Gs]

  f_inhibition1 = (1-f0)*np.reshape(links==-1,[1,Gin+Gs,Gs]) + np.reshape(links!=-1,[1,Gin+Gs,Gs])
  f_inhibition2 = np.prod(f_inhibition1,axis=1) #[BATCH,Gs]

  f = (f_activation2)*f_inhibition2 #[BATCH,Gs]
  return f


def Hill_model_dynamics(b,K,links):
  gamma = 1

  ## inputs=0, 0
  X0 = np.zeros([BATCH,time_points,Gs])
  x0 = 0.1*np.ones([BATCH,Gs])
  for t0 in range(time_points):
    f0 = Hill_model(np.concatenate([np.zeros([BATCH,Gin]),x0],axis=1),b,K,links)
    x0 = (1-gamma*TIMESTEP)*x0 + f0*TIMESTEP
    X0[:,t0,:] = x0
  hit0 = np.var(X0[:,time_points//2:,0],axis=1)<0.002


  ## inputs=0.8, 0
  X1 = np.zeros([BATCH,time_points,Gs])
  x1 = 0.1*np.ones([BATCH,Gs])
  for t1 in range(time_points):
    f1 = Hill_model(np.concatenate([0.8*np.ones([BATCH,1]),np.zeros([BATCH,1]),x1],axis=1),b,K,links)
    x1 = (1-gamma*TIMESTEP)*x1 + f1*TIMESTEP
    X1[:,t1,:] = x1

  grad1= X1[:,time_points//2:,0] - X1[:,(time_points//2-1):-1,0]
  hit1 = (np.var(X1[:,time_points//2:,0],axis=1)>0.01) *\
         ((np.max(grad1,axis=1)*np.min(grad1,axis=1))<-0.001)


  ## inputs=0, 0.8
  X2 = np.zeros([BATCH,time_points,Gs])
  x2 = 0.1*np.ones([BATCH,Gs])
  for t2 in range(time_points):
    f2 = Hill_model(np.concatenate([np.zeros([BATCH,1]),0.8*np.ones([BATCH,1]),x2],axis=1),b,K,links)
    x2 = (1-gamma*TIMESTEP)*x2 + f2*TIMESTEP
    X2[:,t2,:] = x2
  hit2 = (np.mean(X2[:,time_points//2:,0],axis=1)>4*np.mean(X0[:,time_points//2:,0],axis=1)) *\
         (np.var(X2[:,time_points//2:,0],axis=1)<0.002)

  #hit_ = np.random.rand(BATCH)<0.01 # for debug
  hit_ = hit0*hit1*hit2
  
  return X0[hit_,:,:], X1[hit_,:,:], X2[hit_,:,:], hit_
  

def main(_):

  if not os.path.exists(repr(FLAGS.run_num)):
    os.mkdir(repr(FLAGS.run_num))
  if not os.path.exists(repr(FLAGS.run_num)+'/train'):
    os.mkdir(repr(FLAGS.run_num)+'/train')
  if not os.path.exists(repr(FLAGS.run_num)+'/para'):
    os.mkdir(repr(FLAGS.run_num)+'/para')


  for topo_i in range(2**((Gin+Gs)*Gs)):
    links0 = np.array(list(bin(topo_i)[2:]),'float')
    if len(links0) < ((Gin+Gs)*Gs):
       links0 = np.concatenate([np.zeros((Gin+Gs)*Gs-len(links0)), links0], axis=0)
    links = (-1)**np.reshape(links0,[Gin+Gs,Gs])

    #if links[2,1]*links[3,0]<0: # search negative feedback loop only, used for debug
    if True:

      X0s = []
      X1s = []
      X2s = []
      bs  = []
      Ks  = []
      for n in range(20):
        b = np.random.exponential(1,[BATCH,Gin+Gs,Gs])
        K = np.random.exponential(1,[BATCH,Gin+Gs,Gs])
        X0,X1,X2,hit_ = Hill_model_dynamics(b,K,links)
        if len(X0) is not 0:
          X0s.append(X0[:,:,0])
          X1s.append(X1[:,:,0])
          X2s.append(X2[:,:,0])
          bs.append(b[hit_,:,:])
          Ks.append(K[hit_,:,:])
        print('step'+repr(n)+', hit '+repr(len(X0[:,0,0])))
  
      if len(X0s) is not 0:
        bs_ = np.concatenate(bs,axis=0)
        Ks_ = np.concatenate(Ks,axis=0)
        parameters = np.concatenate([np.reshape(bs_,[-1,(Gin+Gs)*Gs]),\
                                     np.reshape(Ks_,[-1,(Gin+Gs)*Gs])], axis=1) #[~,2*(Gin+Gs)*Gs]
        np.savetxt(repr(FLAGS.run_num)+'/para/topology'+repr(topo_i)+'.csv', parameters, fmt='%.4f', delimiter=',')


        X0s_ = np.concatenate(X0s,axis=0)
        X1s_ = np.concatenate(X1s,axis=0)
        X2s_ = np.concatenate(X2s,axis=0)

        plt.subplot(3,1,1)
        plt.plot(np.transpose(X0s_),'-')
        plt.subplot(3,1,2)
        plt.plot(np.transpose(X1s_),'-')
        plt.subplot(3,1,3)
        plt.plot(np.transpose(X2s_),'-')

        plt.savefig(repr(FLAGS.run_num)+'/searching/topology'+repr(topo_i))
        plt.close()
      else:
        print('NULL')
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_num', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
