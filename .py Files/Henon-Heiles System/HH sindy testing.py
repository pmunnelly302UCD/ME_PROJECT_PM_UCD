import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps



def Henon_Heiles(t, x):    
        
    return [
        x[2],  # d(q_1)/dt
        x[3],  # d(q_2)/dt
        -x[0] -2*x[0]*x[1],  # d(p_1)/dt
        -x[1] -x[0]**2 +x[1]**2  # d(p_2)/dt
    ]
    
dt = .01  # Timestep

t_train = np.arange(0, 50, dt)  # Time range to integrate over
x0_train = [0.4, 0.25, 0.25, 0.25]
sol = solve_ivp(Henon_Heiles, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)  # Integrate to produce x(t),y(t),z(t)
x_train = np.transpose(sol.y) 

model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()

# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, 100, dt)  # Longer time range
x0_test = np.array([0.25, 0.25, 0.25, 0.25])  # New initial conditions
sol = solve_ivp(Henon_Heiles, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
x_test = np.transpose(sol.y) 

# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, t=dt))

x_test_sim = model.simulate( x0_test , t_test )
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    #axs[i].set_xlim(24,25)

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
ax1.set(xlabel='x', ylabel='y',
        zlabel='z', title='true simulation')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], 'r--')
ax2.set(xlabel='x', ylabel='y',
        zlabel='z', title='model simulation')
