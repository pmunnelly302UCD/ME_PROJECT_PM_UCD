import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin

#################### Paramaters ####################
A0 = 0.5
A1 = 0.5
OMEGA0 = 1
omega = 1.5
beta = 0.1
    

def Josephson_Junction(t, x):    
        
    return [
        omega,
        x[2],  # d(theta)/dt
        A0/OMEGA0**2 + (A1/OMEGA0**2)*sin((x[0]/OMEGA0)) - sin(x[1]) - (beta/OMEGA0)*x[2] # d(v)/dt
    ]
    
dt = 0.01
t_test = np.arange(0,200,dt)
X0 = [0.5, 0.5]

sol = solve_ivp(Josephson_Junction, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)


 # %% Plotting
 
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
plt.rc('axes', labelsize=15)
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i])
    axs[i].set(xlabel='t', ylabel=['$\phi$', '$\\theta$', 'v'][i])
    #axs[i].set_xlim(150,180)
    #axs[i].legend()
    
    
fig = plt.figure(figsize= (11, 10))
ax1 = fig.add_subplot(111)
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ax1.set(xlabel='$\phi$', '$\\theta$', ylabel='v')

