"""
Created on 25/02/22

Paramater sweep for Lyapunov algorithm on atomic coherence system
(Sweep of Lambda and omega)

Author: Patrick Munnelly

"""

import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns

##############################################################################
# Define initial condition and paramater sweep range values:
    
X0 = [1, 1, 1, 1]

s_omega = np.arange(0.25, 0.26, 0.02)
s_Lambda = np.arange(0, 0.5, 0.04)



##############################################################################
# Define system:

def Henon_Heiles(t, x):    
        
    return [
        x[2],  # d(q_1)/dt
        x[3],  # d(q_2)/dt
        -x[0] -2*x[0]*x[1],  # d(p_1)/dt
        -x[1] -x[0]**2 +x[1]**2  # d(p_2)/dt
    ]

##############################################################################
# Define 2D array to hold maximum Lyapunov exponants:

MLE = np.zeros((s_omega.size, s_Lambda.size)) 

##############################################################################
# Set Lyapunov algorithm paramaters:
epsilon = 0.01  # Pertibation Amplitude
T = 5  # Integral time interval
M = 50 # Integral iterations
N = 4 # Number of state variables in our system
dt = 0.01 # Set timestep for integration

##############################################################################

for i_omega in range(s_omega.size):
     
    print (i_omega)
    
    for i_Lambda in range(s_Lambda.size):
        print ('\t', i_Lambda)
        X0 = [0.25, 0.25, 0.25, 0.25]
        X0[0] = s_omega[i_omega]
        X0[1] = s_Lambda[i_Lambda]
    
        # Now run our Lyapunov algorithm:


        # Perturbed vector:
        x_tilda = np.zeros((N,N))

        # Perturned vector relative to reference vector
        x_tilda_r = np.zeros((N,N))

        # Create initial Orthonormalised perturbed vector:
        p = ([[epsilon, 0, 0, 0],
              [0, epsilon, 0, 0],
              [0, 0, epsilon, 0],
              [0, 0, 0, epsilon]])

        x_tilda_0 = [np.add(X0,p[0]),
                     np.add(X0,p[1]),
                     np.add(X0,p[2]),
                     np.add(X0,p[3])]

        x_tilda_0_r = np.zeros((N,N))

        S = np.zeros(N)

        ##############################################################################
        # Begin main loop:
    #try:    
        for i in range(M):
            # Integrate reference vector over time T:
            sol = solve_ivp(Henon_Heiles, (i*T, (i+1)*T), X0, method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
            X0 = (np.transpose(sol.y))[-1]     
            
            for j in range(N):
                # Integrate each perturbation vector over time T:
                # x_tilda(j) = final value of integral from (x_tilda_0(j)) over T
                sol = solve_ivp(Henon_Heiles, (i*T, (i+1)*T), x_tilda_0[j], method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
                x_tilda[j] = (np.transpose(sol.y))[-1]
                
                # Find the relative vector between each perturbation vector and the refernce vector:
                x_tilda_r[j] = x_tilda[j] - X0
                    
            # Complete a gram schmidt orthogonalization process on relative perturbed vectors:  
            for j in range(N):
                for k in range(j):
                    x_tilda_r[j] = x_tilda_r[j] - (np.dot(x_tilda_r[k], x_tilda_r[j])/np.dot(x_tilda_r[k], x_tilda_r[k])) * x_tilda_r[k]
                    
                # Update the accumulated sums with the new relative vector:
                S[j] = S[j] + np.log(np.linalg.norm(x_tilda_r[j]/epsilon))
                
                x_tilda_0_r[j] = x_tilda_r[j] * epsilon / np.linalg.norm(x_tilda_r[j])
                
                # Compute the absolute vectors for the next iteration:
                x_tilda_0[j] = X0 + x_tilda_0_r[j]
                
        ##############################################################################
        # Calculate final Lyapunov exponant values:
 
        L_exp = S/(M*T)
        
        MLE[i_omega,i_Lambda] = np.max(L_exp)
        
    #except:
        #MLE[i_omega,i_Lambda] = 0
     

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(MLE, linewidth=0.5, xticklabels=np.around(s_Lambda, decimals=2), 
                 yticklabels=np.around(s_omega, decimals=2))
ax.set_xlabel('X0[0]')
ax.set_ylabel('X0[1]')
plt.show()