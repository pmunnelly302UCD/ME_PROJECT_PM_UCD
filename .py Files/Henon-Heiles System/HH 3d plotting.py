import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def Henon_Heiles(t, x):    
        
    return [
        x[2],  # d(q_1)/dt
        x[3],  # d(q_2)/dt
        -x[0] -2*x[0]*x[1],  # d(p_1)/dt
        -x[1] -x[0]**2 +x[1]**2  # d(p_2)/dt
    ]
    
dt = 0.01
t_test = np.arange(0,13,dt)
X0 = [0.25,0.46,0.25,0.25]

sol = solve_ivp(Henon_Heiles, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
plt.rc('axes', labelsize=15)
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i])
    axs[i].set(xlabel='t', ylabel=['$q_1$','$q_2$','$p_1$','$p_2$'][i])
    #axs[i].set_xlim(8,9)
    #axs[i].legend()
    
fig = plt.figure(figsize= (11, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ax1.set(xlabel='$q_1$', ylabel='$q_2$', zlabel='$p_1$')

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(x_test[:, 0], x_test[:, 1], x_test[:, 3])
ax2.set(xlabel='$q_1$', ylabel='$q_2$', zlabel='$_2$')

ax4 = fig.add_subplot(223, projection='3d')
ax4.plot(x_test[:, 0], x_test[:, 2], x_test[:, 3])
ax4.set(xlabel='$q_1$', ylabel='$p_1$', zlabel='$p_2$')

ax7 = fig.add_subplot(224, projection='3d')
ax7.plot(x_test[:, 1], x_test[:, 2], x_test[:, 3])
ax7.set(xlabel='$q_2$', ylabel='$p_1$', zlabel='$p_2$')