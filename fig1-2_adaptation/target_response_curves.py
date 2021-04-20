import numpy as np

def adapt_pulse_double_exp(time_points, height):
  # inputs: scalars
  # output: np.array of shape=time_points
  # double exponential response curve
  # used for results in figs. 1&2
  xs0 = np.linspace(0.0,(time_points-1), time_points)
  y = height*2*(np.exp(-xs0/6) - np.exp(-xs0/3))
  return y

def adapt_pulse_triangular(time_points, height):
  # inputs: scalars
  # output: np.array of shape=time_points
  # triangular pulse for the adaptation task
  # used for results in fig. S1a-c
  xs0 = np.linspace(0.0,(time_points-1), time_points)
  y = np.maximum(np.minimum(0.1*xs0, 1-0.1*xs0), 0)
  return y