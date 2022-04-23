"""
Created on 25/02/22

Paramater sweep for Lyapunov algorithm on 4d-Rosser System
(Sweep of Lambda and omega)

Author: Patrick Munnelly

"""

import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns

##############################################################################
# Define initial condition and paramater sweep range values:
    
X0 = [-10,-6,0,10]

s_omega = np.arange(3, 6, 0.25)
s_Lambda = np.arange(0.32, 0.45, 0.01)

a = 0.25
b = 3 
c = 0.35 
d = 0.05

##############################################################################
# Define system:

def Rossler(t, x):
    return [
        -x[1] -x[2],
        x[0] +a*x[1] +x[3],
        b +x[0]*x[2],
        -c*x[2] +d*x[3]
    ]

##############################################################################
# Define 2D array to hold maximum Lyapunov exponants:

MLE = np.zeros((s_omega.size, s_Lambda.size)) 

##############################################################################
# Set Lyapunov algorithm paramaters:
epsilon = 0.01  # Pertibation Amplitude
T = 5  # Integral time interval
M = 100 # Integral iterations
N = 4 # Number of state variables in our system
dt = 0.01 # Set timestep for integration
limit = 300 # limit for state variable magnitudes

##############################################################################

for i_omega in range(s_omega.size):
    b = s_omega[i_omega]
    print (i_omega)
    for i_Lambda in range(s_Lambda.size):
        c = s_Lambda[i_Lambda]
        print('\t', i_Lambda)
    
        # Now run our Lyapunov algorithm:

        # Reference vector:
        x = X0

        # Perturbed vector:
        x_tilda = np.zeros((N,N))

        # Perturned vector relative to reference vector
        x_tilda_r = np.zeros((N,N))

        # Create initial Orthonormalised perturbed vector:
        p = ([[epsilon, 0, 0, 0],
              [0, epsilon, 0, 0],
              [0, 0, epsilon, 0],
              [0, 0, 0, epsilon]])

        x_tilda_0 = [np.add(x,p[0]),
                     np.add(x,p[1]),
                     np.add(x,p[2]),
                     np.add(x,p[3])]

        x_tilda_0_r = np.zeros((N,N))

        S = np.zeros(N)

        ##############################################################################
        # Begin main loop:
    #try:    
        for i in range(M):
            #print('\t\t', i)
            # Integrate reference vector over time T:
            sol = solve_ivp(Rossler, (i*T, (i+1)*T), x, method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
            x = (np.transpose(sol.y))[-1]     
            if (abs(x[0])>limit or abs(x[1])>limit or abs(x[2])>limit or abs(x[3])>limit):
                break;
                
            for j in range(N):
                # Integrate each perturbation vector over time T:
                # x_tilda(j) = final value of integral from (x_tilda_0(j)) over T
                sol = solve_ivp(Rossler, (i*T, (i+1)*T), x_tilda_0[j], method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
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
        #print(L_exp)
        MLE[i_omega,i_Lambda] = np.max(L_exp)
        
    #except:
        #MLE[i_omega,i_Lambda] = 0
     
# %%

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(MLE, linewidth=0.5, xticklabels=np.around(s_Lambda, decimals=2), 
                 yticklabels=np.around(s_omega, decimals=1))
ax.set_xlabel('c')
ax.set_ylabel('b')
plt.show()