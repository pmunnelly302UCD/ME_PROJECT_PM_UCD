# -*- coding: utf-8 -*-
"""

Created on 05/02/22

Modelling using PyVista

Author: Patrick Munnelly

"""

# Import packages:
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos
from scipy.integrate import solve_ivp
import pyvista as pv


################################################################
# Define time data and initial conditions:
dt = 0.001
t_test = np.arange(0,100,dt)
X0 = [0.5, 0.5, 0.5, 0.5, 1]

################################################################
# Define model system paramaters:
omega = 4
Lambda = 4

################################################################
# Define our model system DEs:
def quantum_model(t, x):    
#    if (x[1]>1): # This bounds phi_e 
 #       x[1] = 0.5
        
    return [
        -omega*Lambda* (x[0]*x[2])**0.5 *cos(x[3])*(sin(x[1])),  # d(I_e)/dt
        omega -0.5*omega*Lambda* (x[2]/x[0])**0.5 *(cos(x[3])*cos(x[1])),  # d(phi_e)/dt
        2*omega*x[4]*Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3])),  # d(I_m)/dt
        omega*x[4]*Lambda* (x[0]/x[2])**0.5 *(cos(x[3])*cos(x[1])),  # d(phi_m)/dt
        -Lambda* (x[0]*x[2])**0.5 *(cos(x[1])*sin(x[3]))  # d(n)/dt
    ]

################################################################
# Solve our equations using solve_ivp:
sol = solve_ivp(quantum_model, (t_test[0], t_test[-1]), X0, method='BDF', t_eval=t_test) # Integrate
x_test = np.transpose(sol.y)

###########################################################################################################
# Create Poincare section of each of our 3D plots
###########################################################################################################
# Set up main figure:
fig = plt.figure(figsize= (11, 10))

################################################################
# 1st plot (I_e, phi_e, I_m)

# Fit a spline to connect our data points and then create a 3D mesh:
spline1 = pv.Spline(x_test[0::20,[0,1,2]],len(x_test))
mesh1 = pv.PolyData(x_test[:,[0,1,2]], lines=spline1.lines)
#mesh1.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices1 = mesh1.slice(normal= 'z', origin=(2,0,2))

# Create a plot:
plotter1 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter1.add_mesh(slices1,color='red')

# Set visualisation paramaters:
plotter1.set_background('white')
plotter1.camera_position='yx'
plotter1.show_grid(color='black', xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='$I_m$')
plotter1.show(screenshot='fig1.png')

# Add to main figure:
fig.add_subplot(331)
plt.imshow(plotter1.image)
plt.title('Plane of $I_e$ VS $\u03C6_e$ VS $I_m$')
plt.axis('off')


################################################################
# 2nd plot (I_e, phi_e, phi_m)

# Fit a spline to connect our data points and then create a 3D mesh:
spline2 = pv.Spline(x_test[0::20,[0,1,3]],len(x_test))
mesh2 = pv.PolyData(x_test[:,[0,1,3]], lines=spline2.lines)
#mesh2.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices2 = mesh2.slice(normal= 'z', origin=(2,0,0))

# Create a plot:
plotter2 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter2.add_mesh(slices2,color='red')

# Set visualisation paramaters:
plotter2.set_background('white')
plotter2.camera_position='yx'
plotter2.show_grid(color='black', xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='$\u03C6_m$')
plotter2.show(screenshot='fig2.png')

# Add to main figure:
fig.add_subplot(332)
plt.imshow(plotter2.image)
plt.title('Plane of $I_e$ vs $\u03C6_e$ vs $\u03C6_m$')
plt.axis('off')

################################################################
# 3rd plot (I_e, phi_e, n)

# Fit a spline to connect our data points and then create a 3D mesh:
spline3 = pv.Spline(x_test[0::20,[0,1,4]],len(x_test))
mesh3 = pv.PolyData(x_test[:,[0,1,4]], lines=spline3.lines)
#mesh3.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices3 = mesh3.slice(normal= 'z', origin=(2,0,0))

# Create a plot:
plotter3 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter3.add_mesh(slices3,color='red')

# Set visualisation paramaters:
plotter3.set_background('white')
plotter3.camera_position='yx'
plotter3.show_grid(color='black', xlabel='$I_e$', ylabel='$\u03C6_e$', zlabel='n')
plotter3.show(screenshot='fig3.png')

# Add to main figure:
fig.add_subplot(333)
plt.imshow(plotter3.image)
plt.title('Plane of $I_e$ vs $\u03C6_e$ vs n')
plt.axis('off')

################################################################
# 4th plot (I_e, I_m, phi_m)

# Fit a spline to connect our data points and then create a 3D mesh:
spline4 = pv.Spline(x_test[0::20,[0,2,3]],len(x_test))
mesh4 = pv.PolyData(x_test[:,[0,2,3]], lines=spline4.lines)
#mesh4.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices4 = mesh4.slice(normal= 'x', origin=(2,2,0))

# Create a plot:
plotter4 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter4.add_mesh(slices4,color='red')

# Set visualisation paramaters:
plotter4.set_background('white')
plotter4.camera_position='zy'
plotter4.show_grid(color='black', xlabel='$I_e$', ylabel='$I_m$', zlabel='$\u03C6_m$')
plotter4.show(screenshot='fig4.png')

# Add to main figure:
fig.add_subplot(334)
plt.imshow(plotter4.image)
plt.title('Plane of $I_e$ vs $I_m$ vs $\u03C6_m$')
plt.axis('off')

################################################################
# 5th plot (I_e, I_m, n)

# Fit a spline to connect our data points and then create a 3D mesh:
spline5 = pv.Spline(x_test[0::20,[0,2,4]],len(x_test))
mesh5 = pv.PolyData(x_test[:,[0,2,4]], lines=spline5.lines)
#mesh5.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices5 = mesh5.slice(normal= 'x', origin=(2,2,0))

# Create a plot:
plotter5 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter5.add_mesh(slices5,color='red')

# Set visualisation paramaters:
plotter5.set_background('white')
plotter5.camera_position='zy'
plotter5.show_grid(color='black', xlabel='$I_e$', ylabel='$I_m$', zlabel='n')
plotter5.show(screenshot='fig5.png')

# Add to main figure:
fig.add_subplot(335)
plt.imshow(plotter5.image)
plt.title('Plane of $I_e$ vs $I_m$ vs n')
plt.axis('off')

################################################################
# 6th plot (I_e, phi_m, n)

# Fit a spline to connect our data points and then create a 3D mesh:
spline6 = pv.Spline(x_test[0::20,[0,3,4]],len(x_test))
mesh6 = pv.PolyData(x_test[:,[0,3,4]], lines=spline6.lines)
#mesh6.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices6 = mesh6.slice(normal= 'x', origin=(2,0,0))

# Create a plot:
plotter6 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter6.add_mesh(slices6,color='red')

# Set visualisation paramaters:
plotter6.set_background('white')
plotter6.camera_position='yz'
plotter6.show_grid(color='black', xlabel='$I_e$', ylabel='$\u03C6_m$', zlabel='n')
plotter6.show(screenshot='fig6.png')

# Add to main figure:
fig.add_subplot(336)
plt.imshow(plotter6.image)
plt.title('Plane of $I_e$ vs $\u03C6_m$ vs n')
plt.axis('off')

################################################################
# 7th plot (phi_e, I_m, phi_m)

# Fit a spline to connect our data points and then create a 3D mesh:
spline7 = pv.Spline(x_test[0::20,[1,2,3]],len(x_test))
mesh7 = pv.PolyData(x_test[:,[1,2,3]], lines=spline7.lines)
#mesh7.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices7 = mesh7.slice(normal= 'x', origin=(0,2,0))

# Create a plot:
plotter7 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter7.add_mesh(slices7,color='red')

# Set visualisation paramaters:
plotter7.set_background('white')
plotter7.camera_position='zy'
plotter7.show_grid(color='black', xlabel='$\u03C6_e$', ylabel='$I_m$', zlabel='$\u03C6_m$')
plotter7.show(screenshot='fig7.png')

# Add to main figure:
fig.add_subplot(337)
plt.imshow(plotter7.image)
plt.title('Plane of $\u03C6_e$ vs $I_m$ vs $\u03C6_m$')
plt.axis('off')

################################################################
# 8th plot (I_e, I_m, n)

# Fit a spline to connect our data points and then create a 3D mesh:
spline8 = pv.Spline(x_test[0::20,[1,2,4]],len(x_test))
mesh8 = pv.PolyData(x_test[:,[1,2,4]], lines=spline8.lines)
#mesh8.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices8 = mesh8.slice(normal= 'x', origin=(0,2,0))

# Create a plot:
plotter8 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter8.add_mesh(slices8,color='red')

# Set visualisation paramaters:
plotter8.set_background('white')
plotter8.camera_position='zy'
plotter8.show_grid(color='black', xlabel='$I_e$', ylabel='$I_m$', zlabel='n')
plotter8.show(screenshot='fig8.png')

# Add to main figure:
fig.add_subplot(338)
plt.imshow(plotter8.image)
plt.title('Plane of $I_e$ vs $I_m$ vs n')
plt.axis('off')

################################################################
# 9th plot (phi_e, phi_m, n)

# Fit a spline to connect our data points and then create a 3D mesh:
spline9 = pv.Spline(x_test[0::20,[1,3,4]],len(x_test))
mesh9 = pv.PolyData(x_test[:,[1,3,4]], lines=spline9.lines)
#mesh9.plot(render_lines_as_tubes=True, line_width=5)
# We define a plane with a normal and an origin:
slices9 = mesh9.slice(normal= 'x', origin=(0,0,0))

# Create a plot:
plotter9 = pv.Plotter(off_screen=True)

# Add our slice to the plot:
plotter9.add_mesh(slices9,color='red')

# Set visualisation paramaters:
plotter9.set_background('white')
plotter9.camera_position='yz'
plotter9.show_grid(color='black', xlabel='$\u03C6_e$', ylabel='$\u03C6_m$', zlabel='n')
plotter9.show(screenshot='fig9.png')

# Add to main figure:
fig.add_subplot(339)
plt.imshow(plotter9.image)
plt.title('Plane of $\u03C6_e$ vs $\u03C6_m$ vs n')
plt.axis('off')

################################################################