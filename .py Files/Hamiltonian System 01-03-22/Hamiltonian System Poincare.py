# -*- coding: utf-8 -*-
"""

Created on 24/03/22

Modelling using PyVista

Author: Patrick Munnelly

"""

# Import packages:
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pyvista as pv


################################################################
# Define time data and initial conditions:
dt = 0.01
t_test = np.arange(0,2000,dt)
X0 = [1.1, 1.2, 1.3, 1.4]

################################################################
# Define model system paramaters:
m1 = 1
m2 = 1
w1 = 1.28
w2 = 1.39
epsilon = 0.415

################################################################
# Define our model system DEs:
def Hamiltonian(t, x):    
        
    return [
        x[2],  # d(q_1)/dt
        x[3],  # d(q_2)/dt
        -(2*epsilon/m1)*x[0]*x[1] - (w1**2)*x[0],  # d(v_1)/dt
        -(epsilon/m2)*((x[0]**2)-(x[1]**2)) - (w2**2)*x[1]  # d(v_1)/dt
    ]
    
################################################################
# Solve our equations using solve_ivp:
sol = solve_ivp(Hamiltonian, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)

###########################################################################################################
# Create Poincare section of each of our 3D plots
###########################################################################################################
# Set up main figure:
fig = plt.figure(figsize= (11, 10))

################################################################
# 1st plot (q1, q2, v1)

# Fit a spline to connect our data points and then create a 3D mesh:
spline1 = pv.Spline(x_test[0::20,[0,1,2]],len(x_test))
mesh1 = pv.PolyData(x_test[:,[0,1,2]], lines=spline1.lines)
#mesh1.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices1 = mesh1.slice(normal= 'x', origin=(0,0,0))

# Create a plot:
plotter1 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter1.add_mesh(slices1,color='red')

# Set visualisation paramaters:
plotter1.set_background('white')
plotter1.camera_position='yz'
plotter1.show_grid(color='black', xlabel='q1', ylabel='q2', zlabel='v1')
plotter1.show(screenshot='fig1.png')

# Add to main figure:
fig.add_subplot(221)
plt.imshow(plotter1.image)
plt.title('Plane of $q_1$ VS $q_2$ VS $v_1$')
plt.axis('off')


################################################################
# 2nd plot (q1, q2, v2)

# Fit a spline to connect our data points and then create a 3D mesh:
spline2 = pv.Spline(x_test[0::20,[0,1,3]],len(x_test))
mesh2 = pv.PolyData(x_test[:,[0,1,3]], lines=spline2.lines)
#mesh2.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices2 = mesh2.slice(normal= 'x', origin=(0,0,0))

# Create a plot:
plotter2 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter2.add_mesh(slices2,color='red')

# Set visualisation paramaters:
plotter2.set_background('white')
plotter2.camera_position='yz'
plotter2.show_grid(color='black', xlabel='q1', ylabel='q2', zlabel='v2')
plotter2.show(screenshot='fig2.png')

# Add to main figure:
fig.add_subplot(222)
plt.imshow(plotter2.image)
plt.title('Plane of $q_1$ VS $q_2$ VS $v_2$')
plt.axis('off')

################################################################
# 3rd plot (q1, v1, v2)

# Fit a spline to connect our data points and then create a 3D mesh:
spline3 = pv.Spline(x_test[0::20,[0,2,3]],len(x_test))
mesh3 = pv.PolyData(x_test[:,[0,2,3]], lines=spline3.lines)
#mesh3.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices3 = mesh3.slice(normal= 'z', origin=(0,0,0))

# Create a plot:
plotter3 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter3.add_mesh(slices3,color='red')

# Set visualisation paramaters:
plotter3.set_background('white')
plotter3.camera_position='yx'
plotter3.show_grid(color='black', xlabel='q1', ylabel='v1', zlabel='v2')
plotter3.show(screenshot='fig3.png')

# Add to main figure:
fig.add_subplot(223)
plt.imshow(plotter3.image)
plt.title('Plane of $q_1$ VS $v_1$ VS $v_2$')
plt.axis('off')

################################################################
# 4th plot (q2, v1, v2)

# Fit a spline to connect our data points and then create a 3D mesh:
spline4 = pv.Spline(x_test[0::20,[1,2,3]],len(x_test))
mesh4 = pv.PolyData(x_test[:,[1,2,3]], lines=spline4.lines)
#mesh4.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices4 = mesh4.slice(normal= 'x', origin=(0,0,0))

# Create a plot:
plotter4 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter4.add_mesh(slices4,color='red')

# Set visualisation paramaters:
plotter4.set_background('white')
plotter4.camera_position='zy'
plotter4.show_grid(color='black', xlabel='q2', ylabel='v1', zlabel='v2')
plotter4.show(screenshot='fig4.png')

# Add to main figure:
fig.add_subplot(224)
plt.imshow(plotter4.image)
plt.title('Plane of $q_2$ VS $v_1$ VS $v_2$')
plt.axis('off')

