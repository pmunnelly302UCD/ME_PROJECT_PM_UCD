"""
Created on 25/02/22

SINDy JJ Lyapunov paramater sweep vs SINDy

Author: Patrick Munnelly

"""

import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.stats import kstest
from numpy import sin

##############################################################################
# Define initial condition and paramater sweep range values:
    
A0 = 0.5
A1 = 0.5
OMEGA0 = 1
omega = 1
beta = 0.01
    
X0 = [0.5, 0.5]

s_omega = np.arange(0.1, 1, 0.05)
s_Lambda = np.arange(0.01, 0.1, 0.005)

##############################################################################
# Set SINDy training and test paramaters:

dt = 0.01 # Set timestep for integration
t_train = np.arange(0, 8, dt)  # Time range to integrate over
x0_train = X0  # Initial Conditions

t_test = np.arange(0, 100, dt)  # Longer time range
x0_test = [0.25, 0.7]  #np.array([8, 7, 15])  # New initial conditions

# Define arrarys to hold model KS score, R^2 score and divergance time
KS = np.zeros((s_omega.size, s_Lambda.size)) 
R_sqr = np.zeros((s_omega.size, s_Lambda.size))
diverge_time = np.full((s_omega.size, s_Lambda.size), t_test[-1]) 

combined_library = ps.FourierLibrary() + ps.IdentityLibrary() # Candidate function library

##############################################################################
# Define system:

def i_ext(t):
    return A0/OMEGA0**2 + (A1/OMEGA0**2)*sin((omega/OMEGA0)*t)    

def Josephson_Junction(t, x):    
    u = i_ext(t)
    return [
        x[1],  # d(theta)/dt
        u - sin(x[0]) - (beta/OMEGA0)*x[1] # d(v)/dt
    ]


##############################################################################
# Define 2D array to hold maximum Lyapunov exponant:
    
MLE = np.zeros((s_omega.size, s_Lambda.size)) 

##############################################################################
# Set Lyapunov algorithm paramaters:

epsilon = 0.01  # Pertibation Amplitude
T = 0.5  # Integral time interval
M = 100 # Integral iterations
N = 2 # Number of state variables in our system
dt = 0.01 # Set timestep for integration
limit = 50000
##############################################################################

for i_omega in range(s_omega.size):
    
    omega = s_omega[i_omega]
    print (i_omega) # Monitor algorthim progession
    
    for i_Lambda in range(s_Lambda.size):
        
        print ('\t', i_Lambda) # Monitor algorthim progession
        beta = s_Lambda[i_Lambda]
    
        # Now run our Lyapunov algorithm:

        # Reference vector:
        x = X0

        # Perturbed vector:
        x_tilda = np.zeros((N,N))

        # Perturned vector relative to reference vector
        x_tilda_r = np.zeros((N,N))

        # Create initial Orthonormalised perturbed vector:
        p = ([[epsilon, 0],
              [0, epsilon]])

        x_tilda_0 = [np.add(X0,p[0]),
                     np.add(X0,p[1])]
                     
        x_tilda_0_r = np.zeros((N,N))

        S = np.zeros(N)

        ##############################################################################
        # Begin main loop:
    #try:    
        for i in range(M):
            # Integrate reference vector over time T:
            sol = solve_ivp(Josephson_Junction, (i*T, (i+1)*T), x, method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
            x = (np.transpose(sol.y))[-1]     
            #if (abs(x[0])>limit or abs(x[1])>limit):
             #   break;
            
            for j in range(N):
                # Integrate each perturbation vector over time T:
                # x_tilda(j) = final value of integral from (x_tilda_0(j)) over T
                sol = solve_ivp(Josephson_Junction, (i*T, (i+1)*T), x_tilda_0[j], method='BDF', t_eval=np.arange(i*T,(i+1)*T,dt))
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
        
        # First create SINDy model
        sol = solve_ivp(Josephson_Junction, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)  # Integrate to produce x(t),y(t),z(t)
        x_train = np.transpose(sol.y)  
        u_train = i_ext(t_train)
        model = ps.SINDy(feature_library=combined_library)
        model.fit(x_train, u=u_train, t=dt)
        
        # Create test trajectory from real system:
        sol = solve_ivp(Josephson_Junction, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
        x_test = np.transpose(sol.y) 
        u_test = i_ext(t_test)
        
        
        # Create SINDy predicted trajectory:
        x_test_sim = model.simulate(x0_test, t_test, u=u_test)
        x_test_sim = np.append(x_test_sim, [x_test_sim[-1]], axis= 0)
        
        # KS Test
        KS1 = (kstest(x_test_sim[:,0], x_test[:,0])).statistic
        KS2 = (kstest(x_test_sim[:,1], x_test[:,1])).statistic
        KS[i_omega,i_Lambda] = (KS1+KS2) / 2
        
        # R^2
        R_sqr[i_omega,i_Lambda] = model.score(x_test, t =dt, u = u_test) 
        if(R_sqr[i_omega,i_Lambda] < 0):
            R_sqr[i_omega,i_Lambda] = 0
        
        # SINDy divergence time
        for i in range(t_test.size):
            diff = np.linalg.norm(x_test[i]-x_test_sim[i])
            if(diff > 0.25*np.linalg.norm(x_test[i])):
                diverge_time[i_omega,i_Lambda] = t_test[i]
                break
        
    #except:
        #MLE[i_omega,i_Lambda] = 0
        #diverge_time[i_omega,i_Lambda] = 0

#%%

fig, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(MLE, linewidth=0.5, xticklabels=np.around(s_Lambda, decimals=2), 
                 yticklabels=np.around(s_omega, decimals=2))
ax.set_xlabel('beta')
ax.set_ylabel('OMEGA0')
ax.set_title('Maximum Lyapunov Exponant')
plt.show()

fig, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(KS, linewidth=0.5, xticklabels=np.around(s_Lambda, decimals=2), 
                 yticklabels=np.around(s_omega, decimals=2))
ax.set_xlabel('beta')
ax.set_ylabel('OMEGA0')
ax.set_title('KS-score')
plt.show()

fig, ax = plt.subplots(figsize=(11, 9))
ax = sns.heatmap(diverge_time, linewidth=0.5, xticklabels=np.around(s_Lambda, decimals=2), 
                 yticklabels=np.around(s_omega, decimals=2))
ax.set_xlabel('beta')
ax.set_ylabel('OMEGA0')
ax.set_title('SINDy divergance time')
plt.show()

# First convert 2D arrays to 1D arrays:
MLE_sorted = MLE[0]
KS_sorted = KS[0]
diverge_time_sorted = diverge_time[0]
R_sqr_sorted = R_sqr[0]

for i in range(s_omega.size-1):
    MLE_sorted = np.append(MLE_sorted, MLE[i+1])
    KS_sorted = np.append(KS_sorted, KS[i+1])
    diverge_time_sorted = np.append(diverge_time_sorted, diverge_time[i+1])
    R_sqr_sorted =  np.append(R_sqr_sorted, R_sqr[i+1])

# Now sort arrays from smallest to largest MLE(bubblesort):    
for i in range(MLE_sorted.size-1, 0, -1):
    for idx in range(i):
        if (MLE_sorted[idx] > MLE_sorted[idx+1]):
            temp1 = MLE_sorted[idx]
            temp2 = KS_sorted[idx]
            temp3 = diverge_time_sorted[idx]
            temp4 = R_sqr_sorted[idx]
            
            MLE_sorted[idx] = MLE_sorted[idx+1]
            KS_sorted[idx] = KS_sorted[idx+1]
            diverge_time_sorted[idx] = diverge_time_sorted[idx+1]
            R_sqr_sorted[idx] = R_sqr_sorted[idx+1]
            
            MLE_sorted[idx+1] = temp1
            KS_sorted[idx+1] = temp2 
            diverge_time_sorted[idx+1] = temp3
            R_sqr_sorted[idx+1] = temp4
            
# Plot result
fig, axs = plt.subplots(figsize=(7, 9))

axs.plot(MLE_sorted, KS_sorted,'.')
axs.set(xlabel='Maximum Lyapunov Exponant', ylabel='Kolmogorov-Smirnov Test Score',
        title = 'Test over 100 seconds, $X_0$ = [0.5,0.5]');
print(np.corrcoef(MLE_sorted, KS_sorted))

# Plot result
fig, axs = plt.subplots(figsize=(7, 9))

axs.plot(MLE_sorted, diverge_time_sorted,'.')
axs.set(xlabel='Maximum Lyapunov Exponant', ylabel='SINDy Prediction Horizon (sec)',
        title = 'Test over 100 seconds, $X_0$ = [0.5,0.5]');
print(np.corrcoef(MLE_sorted, diverge_time_sorted))

# Plot result
fig, axs = plt.subplots(figsize=(7, 9))

axs.plot(MLE_sorted, R_sqr_sorted,'.')
axs.set(xlabel='Maximum Lyapunov Exponant', ylabel='Coefficient of determination ($R^2$)',
        title = 'Test over 100 seconds, $X_0$ = [0.5,0.5]');
print(np.corrcoef(MLE_sorted, R_sqr_sorted))
