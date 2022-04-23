# -*- coding: utf-8 -*-
"""

Created on 05/02/22

Modelling using matplotlib

Author: Patrick Munnelly

"""

# Import packages:
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos
from scipy.integrate import solve_ivp


################################################################
# Define time data and initial conditions:
dt = 0.01
t_test = np.arange(0,2000,dt)
X0 = [0.5, 0.5, 0.5, 0.5, 1]

################################################################
# Define model system paramaters:
omega = 0.5
Lambda = 1.5

################################################################
# Define our model system DEs:
def quantum_model(t, x):    
    if (x[1]>1): # This bounds phi_e 
        x[1] = 0.5
        
    return [
        -omega*Lambda* (x[0]*x[2])**0.5 *cos(x[3])*(sin(x[1])),  # d(I_e)/dt
        omega -0.5*omega*Lambda* (x[2]/x[0])**0.5 *(cos(x[3])*cos(x[1])),  # d(phi_e)/dt
        2*omega*x[4]*Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3])),  # d(I_m)/dt
        omega*x[4]*Lambda* (x[0]/x[2])**0.5 *(cos(x[3])*cos(x[1])),  # d(phi_m)/dt
        -Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3]))  # d(n)/dt
    ]

################################################################
# Solve our equations using solve_ivp:
sol = solve_ivp(quantum_model, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)

################################################################   
# Plot each state variable against time:
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
plt.rc('axes', labelsize=15)
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i])
    axs[i].set(xlabel='t', ylabel=['$I_e$','$\u03C6_e$','$I_m$','$\u03C6_m$', 'n'][i])
   
################################################################   
# 3D plotting of variables against eachother:
fig = plt.figure(figsize= (11, 10))

ax1 = fig.add_subplot(331, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ax1.set(xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='$I_m$')

ax2 = fig.add_subplot(332, projection='3d')
ax2.plot(x_test[:, 0], x_test[:, 1], x_test[:, 3])
ax2.set(xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='$\u03C6_m$')

ax3 = fig.add_subplot(333, projection='3d')
ax3.plot(x_test[:, 0], x_test[:, 1], x_test[:, 4])
ax3.set(xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='n')

ax4 = fig.add_subplot(334, projection='3d')
ax4.plot(x_test[:, 0], x_test[:, 2], x_test[:, 3])
ax4.set(xlabel='$I_e$', ylabel='$I_m$', zlabel='$\u03C6_m$')

ax5 = fig.add_subplot(335, projection='3d')
ax5.plot(x_test[:, 0], x_test[:, 2], x_test[:, 4])
ax5.set(xlabel='$I_e$', ylabel='$I_m$', zlabel='n')

ax6 = fig.add_subplot(336, projection='3d')
ax6.plot(x_test[:, 0], x_test[:, 3], x_test[:, 4])
ax6.set(xlabel='$I_e$', ylabel='$\u03C6_m$', zlabel='n')

ax7 = fig.add_subplot(337, projection='3d')
ax7.plot(x_test[:, 1], x_test[:, 2], x_test[:, 3])
ax7.set(xlabel='$\u03C6_e$', ylabel='$I_m$', zlabel='$\u03C6_m$')

ax8 = fig.add_subplot(338, projection='3d')
ax8.plot(x_test[:, 1], x_test[:, 2], x_test[:, 4])
ax8.set(xlabel='$I_e$', ylabel='$I_m$', zlabel='n')

ax9 = fig.add_subplot(339, projection='3d')
ax9.plot(x_test[:, 1], x_test[:, 3], x_test[:, 4])
ax9.set(xlabel='$\u03C6_e$', ylabel='$\u03C6_m$', zlabel='n')

#ax9 = fig.add_subplot(339, projection='3d')
#ax9.plot(x_test[:, 2], x_test[:, 3], x_test[:, 4])
#ax9.set(xlabel='$I_m$', ylabel='$\u03C6_m$', zlabel='n');

################################################################   
