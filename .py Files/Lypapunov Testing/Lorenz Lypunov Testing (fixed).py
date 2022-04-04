
"""

Created on 15/02/22

Lyapunov algorithm on Lorenz System

Author: Patrick Munnelly

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


##############################################################################
# Define time data and initial conditions:
    
dt = 0.01
t_test = np.arange(0,100,dt)
X0 = [8,7,15]
    
##############################################################################
# Define Lorenz system and paramaters: 

sigma = 10
rho = 28
beta = 8/3

def lorenz(t, x):
    return [
        sigma*(x[1] - x[0]),
        x[0]*(rho - x[2]) - x[1],
        x[0]*x[1] - beta*x[2]
    ]

##############################################################################
# Integrate to get our solution:

sol = solve_ivp(lorenz, (t_test[0], t_test[-1]), X0, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
x_test = np.transpose(sol.y)   
    
##############################################################################
# Plot our solution:
    
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(9,11))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k')
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    
fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
ax1.set(xlabel='$x_0$', ylabel='$x_1$', zlabel='$x_2$',)

##############################################################################
##############################################################################
# Lypounov Calculations
##############################################################################
##############################################################################
# set paramaters:
    
epsilon = 0.01  # Pertibation Amplitude
T = 1  # Integral time interval
M = 100  # Integral iterations
N = 3 # Number of state variables in our system
dt = dt # Change timesetp from earlier in code if needed
##############################################################################
# Set up vectors/matrices

# Reference vector:
x = [8,7,15]

# Perturbed vector:
x_tilda = np.zeros((N,N))

# Perturned vector relative to reference vector
x_tilda_r = np.zeros((N,N))

# Orthonormalised perturbed vector
x_tilda_0 = [[8+epsilon, 7, 15],
             [8, 7+epsilon, 15],
             [8, 7, 15+epsilon]]

x_tilda_0_r = np.zeros((N,N))

S = np.zeros(N)

##############################################################################
# Begin main loop:
    
for i in range(M):
    # Integrate reference vector over time T:
    sol = solve_ivp(lorenz, (i*T, (i+1)*T), x, method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
    x = (np.transpose(sol.y))[-1]     
    
    for j in range(N):
        # Integrate each perturbation vector over time T:
        # x_tilda(j) = final value of integral from (x_tilda_0(j)) over T
        sol = solve_ivp(lorenz, (i*T, (i+1)*T), x_tilda_0[j], method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
        x_tilda[j] = (np.transpose(sol.y))[-1]
        
        # Find the relative vector between each perturbation vector and the refernce vector:
        x_tilda_r[j] = x_tilda[j] - x
            
    # Complete a gram schmidt orthogonalization process on relative perturbed vectors:  
    for j in range(N):
        for k in range(j):
            x_tilda_r[j] = x_tilda_r[j] - (np.dot(x_tilda_r[k], x_tilda_r[j])/np.dot(x_tilda_r[k], x_tilda_r[k])) * x_tilda_r[k]
            
        # Update the accumulated sums with the new relative vector:
        S[j] = S[j] + np.log(np.linalg.norm(x_tilda_r[j]/epsilon))
        
        x_tilda_0_r[j] = x_tilda_r[j] * epsilon / np.linalg.norm(x_tilda_r[j])
        
        # Compute the absolute vectors for the next iteration:
        x_tilda_0[j] = x + x_tilda_0_r[j]
        
##############################################################################
# Calculate final Lyapunov exponant values:
L_exp = S/(M*T)

print(L_exp)






