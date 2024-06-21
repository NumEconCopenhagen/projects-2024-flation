import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import scipy

class ExchangeEcon :
    
    def __init__(self,**kwargs) : # Initiate the model class
        # Make a types.SimpleNamespace (dictionary that allows for dot notation)
        # In other word this is kind of a dictionary with parameters
        par = self.par = SimpleNamespace()

        # Endowments
        par.w1A = 0.8 ; par.w2A = 0.3 # Consumer A's endowment of good 1 and 2
        par.w1B = 1 - par.w1A ; par.w2B = 1 - par.w2A # Consumer B's endowment of good 1 and 2

        # Preference parameters
        par.alpha = 1/3 ; par.beta = 2/3 # Consumer A's and B's income shares for good 1

        # Discrete possibilities (N is set to 76 as to accomodate for zero and 0.5)
        par.N = 76

        
    
    def utility_A(self,x1A,x2A) : # Consumer A's utility function
        return ((x1A)**self.par.alpha)*((x2A)**(1-self.par.alpha))

    def utility_B(self,x1B,x2B) : # Consumer B's utility function
        return ((x1B)**self.par.beta)*((x2B)**(1-self.par.beta))
    
    # Restricted such that one can't demand more than is supplied
    def demand_A(self,p1) : # Consumer A's demand function
        I = self.par.w1A*p1+self.par.w2A
        return self.par.alpha*I/p1, (1-self.par.alpha)*I

    # Restricted such that one can't demand more than is supplied
    def demand_B(self,p1) : # Consumer B's demand function
        I = self.par.w1B*p1+self.par.w2B
        return self.par.beta*I/p1, (1-self.par.beta)*I
    
    def check_market_clearing(self,p1) :
    # This function outputs excess demand in the markets for good 1 and 2
    # Supply is given as the sum of the endowments of each good
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-self.par.w1A + x1B-(1-self.par.w1A)
        eps2 = x2A-self.par.w2A + x2B-(1-self.par.w2A)

        return eps1,eps2

    # Question 1 Hej

    # This function finds all paretoimproving allocations given an endowment.
    def FindCore(self) :
        ''' Finds the paretoimproving allocations in C'''
        
        # x1A and x2A are defined as arrays per the assignments instructions
        x1A_vec = np.linspace(0,1,76) ; x2A_vec = np.linspace(0,1,76) ; 
        
        # The two arrays are combined into a grid
        X,Y = np.meshgrid(x1A_vec,x2A_vec)
        
        # Utility of consumer A and B if they consume their endowment
        uA = self.utility_A(self.par.w1A,self.par.w2A)
        uB = self.utility_B(self.par.w1B,self.par.w2B)
        
        # The conditions are defined per the assignment
        conditionA = self.utility_A(X,Y) - uA >= 0 
        conditionB = self.utility_B(1-X,1-Y) - uB >= 0

        # The conditions are combined
        condition = conditionA & conditionB
        
        # The conditions are applied to the grid
        core = np.dstack((X, Y))[condition]
       
       # To plot we extract two arrays for the x and y axis
        x_core = core[..., 0].flatten()
        y_core = core[..., 1].flatten()
        
        return  x_core,y_core,core
    
    def PlotCore(self,x_core,y_core) :
        ''' Plots the paretoimproving allocations in C'''

        # a. total endowment
        w1bar = 1.0 ; w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")

        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Plot the endowment and core
        ax_A.scatter(self.par.w1A,self.par.w2A,marker='s',color='black',label='endowment',zorder = 2)
        ax_A.scatter(x_core,y_core,ls=':',color='red',label='$\mathcal{C}$',zorder=2,s=10)

        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0))


    # Question 2

    def FindError(self):
        ''' Function to find the error in the market clearing conditions at a given price in the discrete price set. '''
        p1 = np.linspace(0.5, 2.5, self.par.N) # Create the price set
        e1 = np.empty(self.par.N) # Empty vector for epsilon1
        e2 = np.empty(self.par.N) # Empty vector for epsilon2
        
        # Fill the two vectors with the market clearing error
        p_opt = 0 ; eps1_small = 50 ; eps2_small = 50
        for i,p in enumerate(p1):
            e1[i],e2[i] = self.check_market_clearing(p)

            if  np.abs(e1[i]) < np.abs(eps1_small) and np.abs(e2[i]) < np.abs(eps2_small) :
                p_opt = p
                eps1_small = e1[i]
                eps2_small = e2[i]

        print(f'The smallest market clearing error is (eps1,eps2) = ({eps1_small:.3f},{eps2_small:.3f})')
        print(f'The price that has the smallest market clearing error is p1 = {p_opt:.3f}')

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(p1, e1, label='$\epsilon_1$', color='blue',s=10)
        plt.scatter(p1, e2, label='$\epsilon_2$', color='red',s=10)
        plt.axhline(0, color='black', lw=0.5, linestyle='--')
        plt.xlabel('Price of good 1 ($p_1$)')
        plt.ylabel('Market clearing error')
        plt.title('Market clearing error for Goods 1 and 2')
        plt.legend()
        plt.grid(True)

        return p_opt, eps1_small, eps2_small
    

    # Question 3

    def ClearingPrice(self,p1,kappa=0.5,maxiter = 500) :
        ''' This function find the market clearing price '''

        t = 0 # Initiate counter
        
        while True: # Loop to solve for market clearing price
            eps1,eps2 = self.check_market_clearing(p1)

            # Stops the loop when excess demand is close to 0 or if more than 500 iterations has been made
            # Output is printed
            if t >= maxiter :
                print(f'The solver has exceeded {maxiter} iterations')
            elif np.abs(eps1) < 1e-08 :
                print(f'The market clearing price for good 1: {p1:.3f}')
                print(f'Value of the market clearing error for the market of good 1: {eps1:.3f}')
                print(f'Value of the market clearing error for the market of good 2: {eps2:.3f}')
                break   
            
            # If the loop is not stopped the price of good 1 will be updated
            p1 = p1 + kappa*eps1

            # Prints the progress as the iterations go on
            # if t < 5 or t%25 == 0:
                # print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {eps1:14.8f}')
            # elif t == 5:
                # print('   ...')
            t += 1   

        x1As,x2As = self.demand_A(p1) ; uA = self.utility_A(x1As,x2As)
        x1Bs,x2Bs = self.demand_B(p1) ; uB = self.utility_B(x1Bs,x2Bs)
        print(f'The allocation they would end with would be (x1A,x2A) = ({x1As:.2f},{x2As:.2f})')
        print(f'Utility of consumer A in equlibrium is {uA:.3f} and for consumer B it is {uB:.3f}')
        
        return p1

    # Question 4.a

    # Objective function to minimized with regards to the price of good 1
    # Takes a price and calculates the demand of consumer B at that price
    def Objective4(self,p1) :
        x1B,x2B = self.demand_B(p1)
        return -self.utility_A(1-min(x1B,1),1-min(x2B,1))

    def SolveADiscrete(self) :
        ''' Function to find the price and allocation if A chooses in the discrete set'''
        t = 0
        # Vector of possible prices    
        p_vec = np.linspace(0.5,2.5,self.par.N)

        # Find the maximal utility that A can achieve given the demand of B
        p1 = 1 ; uA = 1e-08
        for p in p_vec :
            
            u = -self.Objective4(p)
            if u > uA :
                uA = u
                p1 = p
            
            t += 1
        
        x1Bs,x2Bs = self.demand_B(p1) ; uB = self.utility_B(x1Bs,x2Bs)
        uA = self.utility_A(1-x1Bs,1-x2Bs)
        print(f'The price of good 1 that consumer A would choose: {p1:.3f}')
        print(f'The allocation they would end with would be (x1A,x2A) = ({1-x1Bs:.2f},{1-x2Bs:.2f})')
        print(f'Utility of consumer A at their chosen price is {uA:.3f} and for consumer B it is {uB:.3f}')
        return p1
    
    # Question 4.b

    def SolveAContinous(self) :
        ''' Function to find the price and allocation if A chooses in the continous set'''

        initial_guess = [1] # We need an initial guess for the equilibrium price
        bounds = [(0,None)] # Any positive non-zero price can be chosen

        # Find the maximal utility that A can achieve given the demand of B
        solution = scipy.optimize.minimize(
            self.Objective4, initial_guess, bounds = bounds)
        
        x1Bs,x2Bs = self.demand_B(solution.x) ; uB = self.utility_B(x1Bs,x2Bs)
        uA = self.utility_A(1-x1Bs,1-x2Bs)
        print(f'The price of good 1 that consumer A would choose: {solution.x[0]:.3f}')
        print(f'The allocation they would end with would be (x1A,x2A) = ({1-x1Bs[0]:.2f},{1-x2Bs[0]:.2f})')
        print(f'Utility of consumer A at their chosen price is {uA[0]:.3f} and for consumer B it is {uB[0]:.3f}')
        return solution.x[0]

    
    # Question 5.a

    def MaxUtilACore(self,core) :
        '''Function to maximize consumer A's utility from a given set of allocations'''
        u_max = 1e-08 # Minimum utility value

        # Loop through all pareto improving allocations in C and find the one that maximizes the utility of A
        for xA in core :
            u = self.utility_A(xA[0],xA[1])

            # Save the max utility and allocation
            if u > u_max :
                u_max = u
                x1A = xA[0]
                x2A = xA[1]
        
        # Calculate the utility of consumer B under A's choice
        uB = self.utility_B(1-x1A,1-x2A)

        # Print the results
        print(f'Consumer A chooses the allocation (x1A,x2A) = ({x1A:.2f},{x2A:.2f})')
        print(f'Utility of consumer A at this allocation is {u_max:.3f} and for consumer B it is {uB:.3f}')
        return x1A,x2A

    def MaxUtilParetoImp(self) :
        def objective(x) :
            return -self.utility_A(x[0],x[1])
        
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0],1-x[1])-self.utility_B(self.par.w1B,self.par.w2B)})
        bounds = ((0,1),(0,1))

        initial_guess =[0.8,0.3]

        solution = scipy.optimize.minimize(
            objective, initial_guess, method='SLSQP',
            bounds = bounds, constraints = constraints)
        
        u_max = self.utility_A(solution.x[0],solution.x[1])
        uB = self.utility_B(1-solution.x[0],1-solution.x[1])

        print(f'Consumer A chooses the allocation (x1A,x2A) = ({solution.x[0]:.2f},{solution.x[1]:.2f})')
        print(f'Utility of consumer A at this allocation is {u_max:.3f} and for consumer B it is {uB:.3f}')
        return solution.x[0],solution.x[1]
    
    def SocialPlanner(self) :
        def objective(x) :
            return-(self.utility_A(x[0],x[1])+self.utility_B(1-x[0],1-x[1]))
        
        bounds = ((0,1),(0,1))
        initial_guess = [0.8,0.3]

        solution = scipy.optimize.minimize(
            objective, initial_guess, method='SLSQP',
            bounds = bounds)
        
        uA = self.utility_A(solution.x[0],solution.x[1])
        uB = self.utility_B(1-solution.x[0],1-solution.x[1])
        utot = uA + uB

        print(f'The social planner chooses the allocation (x1A,x2A) = ({solution.x[0]:.3f},{solution.x[1]:.3f})')
        print(f'Utility of consumer A at this allocation is {uA:.3f} and for consumer B it is {uB:.3f}')
        print(f'Total utility becomes {utot:.3f}')

        return solution.x[0],solution.x[1]

    def FindIndifference(self,x1A,x2A) :
        
        x1_vec = np.linspace(1e-08,1,1000)

        uA = self.utility_A(x1A,x2A)
        uB = self.utility_B(1-x1A,1-x2A)

        x2A_vec = np.empty(1000)
        x2B_vec = np.empty(1000)

        for i,x1 in enumerate(x1_vec) :
        # Loops thorugh all values of x1_vec and finds the corresponding values of x2 that secure constant utility for both consumers

            # Local function: When equal to zero we are along the correct indifference curve for consumer B
            def objectiveA(x2) :
                return self.utility_A(x1,x2)-uA
            
            # Local function: When equal to zero we are along the correct indifference curve for consumer B
            def objectiveB(x2) :
                return self.utility_B(x1,x2)-uB

            # For each x1 we find the corresponding x2 value that ensures constant utility along uA and uB
            solA = scipy.optimize.root(objectiveA,0)
            solB = scipy.optimize.root(objectiveB,0)
            
            # Append the solutions to the x2A and x2B vector
            x2A_vec[i] = solA.x[0] 
            x2B_vec[i] = 1-solB.x[0]
        
        # The order of the vector is reversed as the plot is drawn from consumer A's perspective
        x2B_vec = np.flip(x2B_vec)
        
        # We find the two points at which indifference curves crosses, such that we may slice the arrays
        # idx = np.argwhere(np.diff(np.sign(x2A_vec-x2B_vec))).flatten()
        # f = idx[0] + 1 ; l = idx[1] + 1

        # The sliced arrays are outputted
        return x1_vec, x2A_vec, x2B_vec
    
    def PlotSocialPlannerEq(self,x1_vec,x2A_vec,x2B_vec,optA,optB) :
        # a. total endowment
        w1bar = 1.0 ; w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")

        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # Plot the endowment and core
        ax_A.scatter(optA, optB, marker='o',color='blue',label='Social Optimum',zorder = 2)
        ax_A.scatter(self.par.w1A,self.par.w2A,marker='s',color='black',label='endowment',zorder = 2)
        ax_A.plot(x1_vec,x2A_vec,ls=':',color='red',label='$u^A$'f'({optA:.2f},{optB:.2f})',zorder=1)
        ax_A.plot(x1_vec,x2B_vec,ls=':',color='blue',label='$u^B$'f'({1-optA:.2f},{1-optB:.2f})', zorder=1)

        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='upper right',bbox_to_anchor=(1.6,1.0))
