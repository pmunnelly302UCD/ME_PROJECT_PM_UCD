# -*- coding: utf-8 -*-
"""

Created on Sat Nov 13 14:37:49 2021

Autocorrelation of SINDy predicted model on Lorenz System

Topic: SINDy model evaluation
Author: Patrick Munnelly

"""

# Import packages:
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import kstest
import pysindy as ps
import statsmodels.graphics.tsaplots as sgt
#############################################

np.random.seed(100) # Consistency
#############################################
#############################################
# TRAINING/TEST PARAMATER CHOICE:
# Define some script data parameters:
TIMESTEP = 0.001 # Training data timestep
TIME_1 = np.arange(0,10,TIMESTEP) # First training trajectory time vector 
TIME_2 = np.arange(100,110,TIMESTEP) # Second training trajectory time vector
TIME_TEST = np.arange(0,15,TIMESTEP) # Testing data time vector

X0_1 = [1, -1, 1] # Initial conditions for first training trajectory
X0_2 = [-3, 3, -3] # Initial conditions for second training trajectory
X0_TEST = [8, 7, 15] # Initial conditions for testing data

NOISE_LEVEL = 0 # Amplitude of white noise to be added to training data
THRESHOLD = 0.1 # threshold for stlsq optimiser
#############################################
#############################################

# Define Lorenz system paramaters:
sigma = 8
rho = 28
beta = 8/3
#############################################

# Define Lorenz system DEs:
def lorenz(t, x):
    return [
        sigma*(x[1] - x[0]),
        x[0]*(rho - x[2]) - x[1],
        x[0]*x[1] - beta*x[2]
    ]

#############################################   
# Create training data:
dt = TIMESTEP  # Timestep

# First trajectory:
t_train1 = TIME_1  # Time range to integrate over
x0_train1 = X0_1  # Initial conditions
sol1 = solve_ivp(lorenz, (t_train1[0], t_train1[-1]), x0_train1, t_eval=t_train1)  # Integrate to produce x(t),y(t),z(t)
x_train1 = np.transpose(sol1.y)  

# Second trajectory:
t_train2 = TIME_2 # Time range to integrate over
x0_train2 = X0_2  # Initial conditions
sol2 = solve_ivp(lorenz, (t_train2[0], t_train2[-1]), x0_train2, t_eval=t_train2) # Integrate to produce x(t),y(t),z(t)
x_train2 = np.transpose(sol2.y)  

# Add noise to both our trajectories:
x_train1 += np.random.normal(scale = NOISE_LEVEL, size=x_train1.shape) 
x_train2 += np.random.normal(scale = NOISE_LEVEL, size=x_train2.shape) 

# Combine both trajectory data sets into a list:
x_train = [x_train1, x_train2]
#############################################

# Create our SINDy model:
stlsq_opt = ps.STLSQ(threshold = THRESHOLD) # Set threshold
model = ps.SINDy(optimizer=stlsq_opt, feature_names=['x','y','z'])
model.fit(x_train, t=dt, multiple_trajectories=True)
print("\nSINDy constructed model:")
model.print()
##############################################
  
# Evolve the Lorenz equations in time using a different initial condition
t_test = TIME_TEST  # Longer time range
x0_test = X0_TEST  # New initial conditions
sol = solve_ivp(lorenz, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
x_test = np.transpose(sol.y)  

# Compare SINDy-predicted derivatives with finite difference derivatives
print('\nModel score using R^2 test: %f' % model.score(x_test, t=dt))
#############################################

# Simulate forward in time
x_test_sim = model.simulate(x0_test, t_test)

# Plot simlated vs true data;
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    
fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
ax1.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='true simulation')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], 'r--')
ax2.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='model simulation')
#############################################

# Plot the autocorrelation functions for SINDy predicted data and the true data (each dimension seperately):
fig, axs = plt.subplots(3,2, figsize=(14,12))
num_lags = np.arange(0,len(x_test_sim)/10,1)
sgt.plot_acf(x_test_sim[:,0], ax=axs[0,0], lags=num_lags, title='Autocorrelation of SINDy predicted data (x-dimension)')
sgt.plot_acf(x_test[:,0], ax= axs[0,1], lags=num_lags, title='Autocorrelation of real test data (x-dimension)')

sgt.plot_acf(x_test_sim[:,1], ax=axs[1,0], lags=num_lags, title='Autocorrelation of SINDy predicted data (y-dimension)')
sgt.plot_acf(x_test[:,1], ax=axs[1,1], lags=num_lags, title='Autocorrelation of real test data (y-dimension)')

sgt.plot_acf(x_test_sim[:,0], ax=axs[2,0], lags=num_lags, title='Autocorrelation of SINDy predicted data (z-dimension)')
sgt.plot_acf(x_test[:,0], ax=axs[2,1], lags=num_lags, title='Autocorrelation of real test data (z-dimension)')
plt.show()
#############################################

# Run the Kolmogorov-Smirnov test:
print('\nFor x-dimension: ', kstest(x_test_sim[:,0], x_test[:,0]))
print('\nFor y-dimension: ', kstest(x_test_sim[:,1], x_test[:,1]))
print('\nFor z-dimension: ', kstest(x_test_sim[:,2], x_test[:,2]))