import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
from numpy import sin,cos

omega = 1
Lambda = 1

def quantum_model(t, x):    
        
    return [
        -omega*Lambda* (x[0]*x[2])**0.5 *cos(x[3])*(sin(x[1])),  # d(I_e)/dt
        omega -0.5*omega*Lambda* (x[2]/x[0])**0.5 *(cos(x[3])*cos(x[1])),  #d(phi_e)/dt
        2*omega*x[4]*Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3])),  # d(I_m)/dt
        omega*x[4]*Lambda* (x[0]/x[2])**0.5 *(cos(x[3])*cos(x[1])),  # d(phi_m)/dt
        -Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3]))  # d(n)/dt
    ]
    
dt = 0.01  # Timestep

t_train = np.arange(0, 15, dt)  # Time range to integrate over
x0_train = [0.5, 0.5, 0.5, 0.5, 1]
sol = solve_ivp(quantum_model, (t_train[0], t_train[-1]), x0_train, t_eval=t_train)  # Integrate to produce x(t),y(t),z(t)
x_train = np.transpose(sol.y) 

# Create custom function library:
my_functions = [
    lambda x,y : (x*y)**0.5,
    lambda x,y : (x/y)**0.5,
    lambda x,y : (sin(x-y)),
    lambda x,y : (sin(x+y)),
    lambda x,y : (cos(x-y)),
    lambda x,y : (cos(x+y))
]
my_function_names = [
    lambda x,y : 'sqrt(' + x + y + ')',
    lambda x,y : 'sqrt(' + x + '/' + y + ')',
    lambda x,y : 'sin(' + x + '-' + y + ')',
    lambda x,y : 'sin(' + x + '+' + y + ')',
    lambda x,y : 'cos(' + x + '-' + y + ')',
    lambda x,y : 'cos(' + x + '+' + y + ')'
]
custom_library = ps.CustomLibrary(
    library_functions=my_functions, function_names=my_function_names)

# Library Selection:
combined_library = ps.GeneralizedLibrary([
    ps.PolynomialLibrary(), ps.FourierLibrary()],
    tensor_array = [[1, 1]])

# Optimiser Selection:
THRESHOLD = 0.1 # threshold for stlsq optimiser
stlsq_opt = ps.STLSQ(threshold = THRESHOLD) # Set threshold


model = ps.SINDy(feature_library = combined_library)
model.fit(x_train, t=dt)
model.print()

# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, 50, dt)  # Longer time range
x0_test = [0.5, 0.5, 0.5, 0.5, 1]
sol = solve_ivp(quantum_model, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
x_test = np.transpose(sol.y) 

# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, t=dt))

x_test_sim = model.simulate( x0_test , t_test )

#%%
fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    #axs[i].set_xlim(24,25)

#ig = plt.figure(figsize=(10, 4.5))
#ax1 = fig.add_subplot(121, projection='3d')
#ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], 'k')
#ax1.set(xlabel='x', ylabel='y',
#        zlabel='z', title='true simulation')

#ax2 = fig.add_subplot(122, projection='3d')
#ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], 'r--')
#ax2.set(xlabel='x', ylabel='y',
#        zlabel='z', title='model simulation')
