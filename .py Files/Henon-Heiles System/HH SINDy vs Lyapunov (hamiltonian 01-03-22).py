"""
Created on 25/02/22

SINDy
(Sweep of Lambda and omega)

Author: Patrick Munnelly

"""

import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
import pysindy as ps
import matplotlib.pyplot as plt

##############################################################################
# Define initial condition and paramater sweep range values:
    
X0 = [0.25, 0.25, 0.25, 0.25]

s_omega = np.arange(0.25, 0.26, 0.02)
s_Lambda = np.arange(0, 0.5, 0.005)

#s_omega = np.concatenate((np.arange(1.28, 1.4, 0.01), np.arange(1.4, 1.8, 0.1)), axis=0)
#s_Lambda = np.concatenate((np.arange(1.28, 1.4, 0.01), np.arange(1.4, 1.8, 0.1)), axis=0)

#%%

print(0.5*(X0[2]**2 + X0[3]**2) + 0.5*(s_Lambda**2 + X0[1]**2 + 2*((s_Lambda**2)*X0[1] - (2/3)*X0[1]**3)))

#%%

##############################################################################
# Set SINDy training and test paramaters:

dt = 0.01 # Set timestep for integration
t_train = np.arange(0, 7, dt)  # Time range to integrate over
x0_train = X0  # Initial Conditions

t_test = np.arange(0, 150, dt)  # Longer time range
x0_test = X0 #np.array([8, 7, 15])  # New initial conditions

diverge_time = np.full((s_omega.size, s_Lambda.size), t_test[-1]) 
R_sqr = np.zeros((s_omega.size, s_Lambda.size)) 
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
    
    print (i_omega) # Monitor algorthim progession
    
    for i_Lambda in range(s_Lambda.size):
        
        print ('\t', i_Lambda) # Monitor algorthim progession
        
        X0 = [0.25, 0.25, 0.25, 0.25]
        X0[1] = s_omega[i_omega]
        X0[0] = s_Lambda[i_Lambda]
    
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
            # Integrate reference vector over time T:
            sol = solve_ivp(Henon_Heiles, (i*T, (i+1)*T), x, method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
            x = (np.transpose(sol.y))[-1]     
            
            for j in range(N):
                # Integrate each perturbation vector over time T:
                # x_tilda(j) = final value of integral from (x_tilda_0(j)) over T
                sol = solve_ivp(Henon_Heiles, (i*T, (i+1)*T), x_tilda_0[j], method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
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
        
        MLE[i_omega,i_Lambda] = np.max(L_exp)
        
        ##############################################################################
        # Create SINDy model and calculate divergance time with true system
        
        x0_train = X0  # Initial Conditions
        x0_test = X0 #np.array([8, 7, 15])  # New initial conditions
        
        # First create SINDy model
        sol = solve_ivp(Henon_Heiles, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)  # Integrate to produce x(t),y(t),z(t)
        x_train = np.transpose(sol.y)  
        model = ps.SINDy()
        model.fit(x_train, t=dt)
        
        # Create test trajectory from real system:
        sol = solve_ivp(Henon_Heiles, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
        x_test = np.transpose(sol.y) 
        
        R_sqr[i_omega,i_Lambda] = model.score(x_test, t =dt) 
        if(R_sqr[i_omega,i_Lambda] < 0):
            R_sqr[i_omega,i_Lambda] = 0
        # Create SINDy predicted trajectory:
        x_test_sim = model.simulate(x0_test, t_test)
        
        for i in range(t_test.size):
            diff = np.linalg.norm(x_test[i]-x_test_sim[i])
            if (diff > 0.25*np.linalg.norm(x_test[i])):
                diverge_time[i_omega,i_Lambda] = t_test[i]
                break
        
    #except:
        #MLE[i_omega,i_Lambda] = 0
        #diverge_time[i_omega,i_Lambda] = 0


#%% Plotting

# For Lyapunov Exponent:
fig, ax = plt.subplots(figsize=(11, 3))
ax = sns.heatmap(MLE, linewidth=0.5, xticklabels=np.around(E, decimals=3), 
                 yticklabels=np.around(s_omega, decimals=1))
ax.set_xlabel('$X_0[0]$')
ax.set_ylabel('')
ax.set_title('MLE')
plt.show()

# For divergance time
fig, ax = plt.subplots(figsize=(11, 3))
ax = sns.heatmap(diverge_time, linewidth=0.5, xticklabels=np.around(E, decimals=3),
                 yticklabels=np.around(s_omega, decimals=1))
ax.set_xlabel('$X_0[0]$')
ax.set_ylabel('')
ax.set_title('Divergance time')
plt.show()

# For R_sqr
fig, ax = plt.subplots(figsize=(11, 3))
ax = sns.heatmap(R_sqr, linewidth=0.5, xticklabels=np.around(E, decimals=3), 
                 yticklabels=np.around(s_omega, decimals=1))
ax.set_xlabel('$X_0[0]$')
ax.set_ylabel('')
ax.set_title('$R^2$')
plt.show()

# First convert 2D arrays to 1D arrays:
MLE_sorted = MLE[0]
diverge_time_sorted = diverge_time[0]
R_sqr_sorted = R_sqr[0]

##############################################################
# MLE VS SINDy divergance time
for i in range(s_omega.size-1):
    MLE_sorted = np.append(MLE_sorted, MLE[i+1])
    diverge_time_sorted = np.append(diverge_time_sorted, diverge_time[i+1])
    R_sqr_sorted =  np.append(R_sqr_sorted, R_sqr[i+1])

# Now sort arrays from smallest to largest MLE(bubblesort):    
for i in range(MLE_sorted.size-1, 0, -1):
    for idx in range(i):
        if (MLE_sorted[idx] > MLE_sorted[idx+1]):
            temp1 = MLE_sorted[idx]
            temp2 = diverge_time_sorted[idx]
            temp3 = R_sqr_sorted[idx]
            
            MLE_sorted[idx] = MLE_sorted[idx+1]
            diverge_time_sorted[idx] = diverge_time_sorted[idx+1]
            R_sqr_sorted[idx] = R_sqr_sorted[idx+1]
            
            MLE_sorted[idx+1] = temp1
            diverge_time_sorted[idx+1] = temp2
            R_sqr_sorted[idx+1] = temp3
            
# Plot result MLE vs divergence time
fig, axs = plt.subplots(figsize=(7, 9))

axs.plot(MLE_sorted, diverge_time_sorted,'.')
axs.set(xlabel='Maximum Lyapunov Exponant', ylabel='SINDy Prediction Horizon (sec)',
        title = 'Test over 50 seconds');

# Plot result MLE vs R^2
fig, axs = plt.subplots(figsize=(7, 9))

axs.plot(MLE_sorted, R_sqr_sorted,'.')
axs.set(xlabel='Maximum Lyapunov Exponant', ylabel='Coefficient of determination ($R^2$)',
        title = 'Test over 50 seconds');
print(np.corrcoef(MLE_sorted, R_sqr_sorted))
