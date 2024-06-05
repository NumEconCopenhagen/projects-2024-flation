from types import SimpleNamespace
import numpy as np
from scipy import optimize

class InFiniteHorizonStandard():

    def __init__(self):
        ''' Initialize the InFiniteHorizonStandard model'''

        self.par = SimpleNamespace() # SimpleNamespace for parameters
        self.ss = SimpleNamespace() # SimpleNamespace for steady state values
        self.path = SimpleNamespace() # SimpleNamespace for transition paths

        self.setup() # Call the setup function during initialization
        self.allocate() # Call the allocate function during during initialization


    def setup(self):
        ''' Defines the parameters and initial values of the model'''

        par = self.par # Call the parameter SimpleNamespace

        # Parameters of the household
        par.sigma = 2 # Coefficient of relative risk aversion
        par.beta = np.nan # Rate of time preference (Note to self: it is set at 0.98 in the article)

        # Parameters of the firm
        par.A = 1 # Total factor productivity
        par.B = 1 # Productivity of physical capital
        par.eta = 0.66 # Labor weight in CES production function
        par.psi = 0.05  # Substitution parameter in CES production function
        par.delta = 0.1 # Depreciation rate on physical capital

        # Initial values and length of transition path
        par.k_lag_0 = 1.0
        par.T = 300


    def allocate(self):
        ''' Allocates arrays for the variables that transition towards steady state '''

        par = self.par
        path = self.path

        endogenous = ['k', # Capital per worker in period t
                      'k_lag', # Captial per worker in period t-1
                      'c', # Consumption per worker in period t
                      'rk', # Return on capital in period t
                      'r', # Real rate of return in period t
                      'w', # Labor wage in period t
                      'y', # Output in period t
                      'i'] # Investments in period t
        
        for var in endogenous :
            path.__dict__[var] = np.nan * np.ones(par.T)
            # Numpy array of T nans for each of the endogenous variables


    def CES_production(self,k_lag) :
        ''' The firms production function '''

        par = self.par

        # Composite input of the production function
        composite = par.eta * k_lag ** par.psi + (1-par.eta) * (1.0) ** par.psi

        # Total output of the producation function
        y = par.A * composite ** (1/par.psi)
        
        # Return on capital
        rk = par.A * composite ** ((1-par.psi)/par.psi) * par.eta * k_lag ** (par.psi-1)
        
        # Labor wages
        w = par.A * composite ** ((1-par.psi)/par.psi) * (1-par.eta) * k_lag ** (par.psi-1)

        return y, rk, w
    
    def steady_state(self, ky_ss) :

        par = self.par
        ss = self.ss

        ss.k = ky_ss
        # y,_,_ = self.CES_production(ss.k) 

        ss.y, ss.rk, ss.w = self.CES_production(ss.k)
        # assert np.isclose(ss.y,1.0)  



