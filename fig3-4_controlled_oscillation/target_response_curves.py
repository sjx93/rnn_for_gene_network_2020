import numpy as np
  
def spikes(time_points, position,height,duration,):
  # inputs: np.array of position,height,duration for each triangular pulse
  # arbitrary number of pulses
  # pulse positions arranged in ascending order
  # output: np.array of shape=time_points
  # used for results in fig. 3-4

  num_peaks = position.shape[0]
  xs0 = np.linspace(0.0,(time_points-1), time_points)
  xs = np.tile(np.expand_dims(xs0,0),[num_peaks,1]) #[num_peaks, time_points]
  y0 = np.expand_dims((position/duration + 1)*2*height,1) - 2*np.expand_dims(height/duration,1)*xs
  y0_= np.expand_dims(height,1)-np.abs(y0-np.expand_dims(height,1))
  position_next = np.concatenate([position[1:],time_points+np.ones(1)],axis=0) #[num_peaks]
  mask = np.float32((xs>=np.expand_dims(position,1))*\
                    (xs<np.expand_dims(position_next,1))*\
                    (xs<np.expand_dims(position+duration,1)))
  y1 = y0_*mask #[num_peaks, time_points]
  y = np.float32(np.sum(y1,axis=0))
  return y