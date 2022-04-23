"""
Created on 25/02/22

SINDy Testing for JJ model

Author: Patrick Munnelly

"""

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps
import matplotlib.pyplot as plt
from numpy import sin

##############################################################################
# Define initial condition and paramater sweep range values:
    
A0 = 0.5
A1 = 0.5
OMEGA0 = 0.3
omega = 1.3
beta = 0.17
    
X0 = [0, 1, 1]

##############################################################################
# Set SINDy training and test paramaters:

dt = 0.01 # Set timestep for integration
t_train = np.arange(0, 20, dt)  # Time range to integrate over
x0_train = X0  # Initial Conditions

t_test = np.arange(0, 50, dt)  # Longer time range
x0_test = X0 #np.array([8, 7, 15])  # New initial conditions

combined_library = ps.FourierLibrary() + ps.IdentityLibrary() # Candidate function library

##############################################################################
# Define system:
def Josephson_Junction(t, x):    
        
    return [
        omega,
        x[2],  # d(theta)/dt
        A0/OMEGA0**2 + (A1/OMEGA0**2)*sin((x[0]/OMEGA0)) - sin(x[1]) - (beta/OMEGA0)*x[2] # d(v)/dt
    ]

# Create SINDy model and compare to true system

# First create SINDy model
sol = solve_ivp(Josephson_Junction, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)  # Integrate to produce x(t),y(t),z(t)
x_train = np.transpose(sol.y)  
model = ps.SINDy(feature_library=combined_library)
model.fit(x_train, t=dt)
model.print()

# Create test trajectory from real system:
sol = solve_ivp(Josephson_Junction, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
x_test = np.transpose(sol.y) 


# Create SINDy predicted trajectory:
x_test_sim = model.simulate(x0_test, t_test)
#x_test_sim = np.append(x_test_sim, [x_test_sim[-1]], axis= 0)
#for i in range(x_test.shape[1]):
#    x_test_sim[:,i] = np.append(x_test_sim[:,i], x_test_sim[-1,i])

# %% Plotting 

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:,i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    axs[i].set_xlim(0,10)