import argparse
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


FLAGS = None
BATCH = 16
Gs = 10 # num of dynamic dims
Gin= Gs # Totalistic CA, neighborhood state
Nx = 20+1 # num of cells
TIMESTEP = 0.1
time_points = 400

def Hill_model(x,b,b0,K,links):
  # x, current input and state [BATCH,Nx,Gin+Gs]
  # b, max_rate for Hill activation term [BATCH,Gin+Gs,Gs]
  # b0,base level expression [BATCH,Gs]
  # K, Micheales constant [BATCH,Gin+Gs, Gs]
  # links, network topology +1/0/-1, [Gin+Gs, Gs]

  Hill_n = 2
  f0 = (np.reshape(x,[BATCH,Nx,Gin+Gs,1])**Hill_n)/ \
       (np.reshape(K,[BATCH,1,Gin+Gs,Gs])**Hill_n + np.reshape(x,[BATCH,Nx,Gin+Gs,1])**Hill_n) #[BATCH,Nx,Gin+Gs,Gs]

  f_activation1 = f0*np.reshape(b,[BATCH,1,Gin+Gs,Gs])*np.reshape(links==1,[1,1,Gin+Gs,Gs])
  f_activation2 = np.sum(f_activation1,axis=2) #[BATCH,Nx,Gs]

  f_inhibition1 = (1-f0)*np.reshape(links==-1,[1,1,Gin+Gs,Gs]) + np.reshape(links!=-1,[1,1,Gin+Gs,Gs])
  f_inhibition2 = np.prod(f_inhibition1,axis=2) #[BATCH,Nx,Gs]

  #f = (f_activation2 + np.reshape(b0,[BATCH,1,Gs]))*f_inhibition2 #[BATCH,Nx,Gs]
  f = (f_activation2)*f_inhibition2 #[BATCH,Nx,Gs]
  return f


def Hill_model_dynamics(b,b0,K,links):
  gamma = 1
  X0 = np.zeros([BATCH,time_points,Nx,Gs])

  x0 = np.zeros([BATCH,Nx,Gs])
  x0[:,(Nx//2),:] = np.float32(np.random.rand(BATCH,Gs)<0.5)

  for t0 in range(time_points):
    x0_neighbor = 0.5*np.concatenate([np.reshape(x0[:,Nx-1,:],[BATCH,1,Gs]), x0[:,0:(Nx-1),:]], axis=1) +\
                  0.5*np.concatenate([x0[:,1:,:], np.reshape(x0[:,0,:],[BATCH,1,Gs])], axis=1)

    f0 = Hill_model(np.concatenate([x0_neighbor,x0],axis=2),b,b0,K,links) +\
         np.random.normal(0,0.0001,[BATCH,Nx,Gs])
    x0 = (1-gamma*TIMESTEP)*x0 + f0*TIMESTEP #[BATCH,Nx,Gs]
    X0[:,t0,:,:] = x0

    hit_1 = np.mean(np.std(X0[:,3*(time_points//4):,:,:], axis=1),axis=(1,2)) > 0.2 # temporal dynamic
    hit_2 = np.mean(np.std(X0[:,3*(time_points//4):,:,:], axis=2),axis=(1,2)) > 0.2 # spatial pattern
    hit_3 = ( np.mean((X0[:,:,0:(Nx//2),:] - X0[:,:,(Nx//2+1):,:])**2, axis=(1,2,3))/\
              np.mean(X0**2, axis=(1,2,3)) ) < 0.1 # noise robustness
    hit_ = hit_1*hit_2
  return X0[hit_,:,:,:], hit_
  

def main(_):

  if not os.path.exists('CA_ground_truth_models'):
    os.mkdir('CA_ground_truth_models')
  if not os.path.exists('CA_ground_truth_models/imgs'):
    os.mkdir('CA_ground_truth_models/imgs')
  if not os.path.exists('CA_ground_truth_models/para'):
    os.mkdir('CA_ground_truth_models/para')


  parameter_writer = csv.writer(open('CA_ground_truth_models/para/parameters.csv', 'w'))
  j = 0
  for n in range(2000):
    links = (2*np.float32(np.random.rand(Gin+Gs,Gs)<0.5)-1)*np.float32(np.random.rand(Gin+Gs,Gs)<0.7)
    print('trying_network_topology, '+repr(n))
    b = 1+0*np.random.rand(BATCH,Gin+Gs,Gs)
    b0= 0*np.random.rand(BATCH,Gs)
    K = np.random.rand(BATCH,Gin+Gs,Gs)

    X0_, hit_ = Hill_model_dynamics(b,b0,K,links)

    if len(X0_) is not 0:
      b_hit = b[hit_,:,:]
      Ks_hit = K[hit_,:,:]

      for i_hit in range(X0_.shape[0]):
        plt.imshow(X0_[i_hit,range(0,time_points,5),:,0:3])
        plt.savefig('CA_ground_truth_models/imgs/model_'+repr(j))

        links_j = np.reshape(links,[(Gin+Gs)*Gs])
        b_j = np.reshape(b_hit[i_hit,:,:],[(Gin+Gs)*Gs])
        K_j = np.reshape(Ks_hit[i_hit,:,:],[(Gin+Gs)*Gs])
        parameters_j = np.concatenate([b_j, K_j, links_j], axis=0)
        parameter_writer.writerow(parameters_j)
        j += 1

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_name', type=int, default=1, help='***')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
