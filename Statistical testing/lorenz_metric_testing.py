"""
Regression Score Metric Testing on Lorenz System

Topic: SINDy model evaluation
Author: Patrick Munnelly

"""

# Import packages:
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import pandas as pd
import sklearn as skl
#############################################

np.random.seed(100) # Consistency

# Define a list of regression metrics to be tested:
metrics = ['explained_variance_score', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
           'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
           'r2', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error']
#############################################

# Define some script parameters:
ROWS = 15  # Define the number of different Lorenz paramater combinatons to be tested
JUMP = 5 # How much to increment lorenz 

TIMESTEP = 0.001 # Training data timestep
TIME_1 = np.arange(0,10,TIMESTEP) # First training trajectory time vector 
TIME_2 = np.arange(100,110,TIMESTEP) # Second training trajectory time vector
TIME_TEST = np.arange(0,15,TIMESTEP) # Testing data time vector

X0_1 = [1, -1, 1] # Initial conditions for first training trajectory
X0_2 = [-3, 3, -3] # Initial conditions for second training trajectory
X0_TEST = [8, 7, 15] # Initial conditions for testing data

NOISE_LEVEL = 0.05 # Amplitude of white noise to be added to training data
THRESHOLD = 0.1 # threshold for stlsq optimiser
#############################################

# Define our dataframe to hold results:
df = pd.DataFrame(columns=metrics)
#############################################

# Define Lorenz system paramaters:
sigma = 1
rho = 1
beta = 1
#############################################

# for loop to test different Lorenz system paramater values:
for j in range(0,ROWS+1):
    if (j <= ROWS/3):
        sigma+=JUMP
    elif(j <= 2*ROWS/3):
        rho+=JUMP
    else:
        beta+=JUMP
    # Define Lorenz system DEs:
    def lorenz(t, x):
        return [
            sigma*(x[1] - x[0]),
            x[0]*(rho - x[2]) - x[1],
            x[0]*x[1] - beta*x[2]
        ]

   
    # Create training data:
    dt = TIMESTEP  # Timestep

    # First trajectory:
    t_train1 = TIME_1  # Time range to integrate over
    x0_train1 = X0_1  # Initial conditions
    sol1 = solve_ivp(lorenz, (t_train1[0], t_train1[-1]), x0_train1, t_eval=t_train1)  # Integrate to produce x(t),y(t),z(t)
    x_train1 = np.transpose(sol1.y)  
    
    # Second trajectory:
    t_train2 = TIME_2 # Time range to integrate over
    x0_train2 = X0_2  # Initial conditions
    sol2 = solve_ivp(lorenz, (t_train2[0], t_train2[-1]), x0_train2, t_eval=t_train2) # Integrate to produce x(t),y(t),z(t)
    x_train2 = np.transpose(sol2.y)  
    
    # Add noise to both our trajectories:
    x_train1 += np.random.normal(scale = NOISE_LEVEL, size=x_train1.shape) 
    x_train2 += np.random.normal(scale = NOISE_LEVEL, size=x_train2.shape) 
    
    # Combine both trajectory data sets into a list:
    x_train = [x_train1, x_train2]
    #
    
    # Create our SINDy model:
    stlsq_opt = ps.STLSQ(threshold = THRESHOLD) # Set threshold
    model = ps.SINDy(optimizer=stlsq_opt)
    model.fit(x_train, t=dt, multiple_trajectories=True)
    #
      
    # Evolve the Lorenz equations in time using a different initial condition
    t_test = TIME_TEST  # Longer time range
    x0_test = X0_TEST  # New initial conditions
    sol = solve_ivp(lorenz, (t_test[0], t_test[-1]), x0_test, t_eval=t_test) # Integrate to produce x(t),y(t),z(t)
    x_test = np.transpose(sol.y)  
    # Compare SINDy-predicted derivatives with finite difference derivatives
    # We now loop to test each metric: 
    #for i in range(0, len(metrics)):
    df.at[j,metrics[0]] = model.score(x_test, t=dt, metric = skl.metrics.mean_absolute_percentage_error)
    df[metrics[0]].to_csv('script results', index=False, float_format=':,.2%')