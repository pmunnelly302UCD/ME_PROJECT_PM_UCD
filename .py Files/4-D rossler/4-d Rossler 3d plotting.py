import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

a = 0.25
b = 4.5
c = 0.4
d = 0.05

def Rossler(t, x):
    return [
        -x[1] -x[2],
        x[0] +a*x[1] +x[3],
        b +x[0]*x[2],
        -c*x[2] +d*x[3]
    ]
    
dt = 0.01
t_test = np.arange(0,60,dt)
X0 = [-10,-6,0,10]

sol = solve_ivp(Rossler, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7,9))
plt.rc('axes', labelsize=15)
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i])
    axs[i].set(xlabel='t', ylabel=['x','y','z','w'][i])
    #axs[i].set_xlim(8,9)
    #axs[i].legend()
    
fig = plt.figure(figsize= (11, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2])
ax1.set(xlabel='$q_1$', ylabel='$q_2$', zlabel='$v_1$')
ax1.view_init(30,210)

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(x_test[:, 0], x_test[:, 1], x_test[:, 3])
ax2.set(xlabel='$q_1$', ylabel='$q_2$', zlabel='$v_2$')

ax4 = fig.add_subplot(223, projection='3d')
ax4.plot(x_test[:, 0], x_test[:, 2], x_test[:, 3])
ax4.set(xlabel='$q_1$', ylabel='$v_1$', zlabel='$v_2$')

ax7 = fig.add_subplot(224, projection='3d')
ax7.plot(x_test[:, 1], x_test[:, 2], x_test[:, 3])
ax7.set(xlabel='$q_2$', ylabel='$v_1$', zlabel='$v_2$')