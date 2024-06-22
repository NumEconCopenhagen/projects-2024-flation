from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class Model() :
    def __init__(self, alpha, s_k, s_h, s_m, delta, A_H, A_L, eta, theta):
        self.alpha = alpha
        self.s_k = s_k
        self.s_h = s_h
        self.s_m = s_m
        self.delta = delta
        self.A_H = A_H
        self.A_L = A_L
        self.eta = eta
        self.theta = theta


    def output_and_accumulation(self, k, h, psi) :
        """ Computes the output and the accumulation functions """

        # The composite input is the the part of the CES production function in parantheses
        # It is calculated seperetaly for easier reading
        composite_input = (self.A_H * h)**psi + (self.A_L * (1 - h))**psi
        
        # Output per capita
        Y = k**self.alpha * composite_input**((1 - self.alpha) / psi)  # Corrected use of psi
        
        # Capital accumulation per capita
        k_next = self.s_k * Y + (1 - self.delta) * k - k
        
        # Skilled labor accumulation per capita
        h_next = self.s_h * Y + (1 - self.delta) * h - h
        
        return k_next, h_next


    def find_steady_state(self, k_init, h_init, psi):
        """ Function to find the steady state value given initial guesses """
        
        # We use the root function from scipy to find the steady state
        solution = optimize.root(lambda vars: self.output_and_accumulation(*vars, psi), [k_init, h_init], method='hybr')
        
        # In case the optimiziation doesn't converge we show an error
        if not solution.success:
            print(f"Optimization did not converge: {solution.message}")
        
        return solution.x


    def plot_phase_diagram(self, k_range, h_range, psi):
        """ Function to plot the phase diagram given a value of psi """

        # Make a grid from the two ranges
        K, H = np.meshgrid(k_range, h_range)
        
        # 
        results = np.vectorize(lambda k, h: self.output_and_accumulation(k, h, psi),signature='(),()->(),()')(K, H)
        
        
        DeltaK = results[0]
        
        
        DeltaH = results[1]


        plt.figure(figsize=(8, 6))

        # We plot the phase diagram with level curves
        contours_k = plt.contour(K, H, DeltaK, levels=[0], colors='blue', linestyles='solid')
        plt.clabel(contours_k, inline=True, fontsize=8, fmt={0: 'Δk = 0'})

        contours_h = plt.contour(K, H, DeltaH, levels=[0], colors='red', linestyles='solid')
        plt.clabel(contours_h, inline=True, fontsize=8, fmt={0: 'Δh = 0'})

        plt.title(f'Phase Diagram with ψ = {psi}')
        plt.xlabel('Capital per capita (k)')
        plt.ylabel('Skilled labor per capita (h)')
        plt.grid(True)
        # plt.legend()
        plt.show()

    def compute_equilibrium_ext(self, k, h, m, psi):
        """ Computes the equilibrium condition for given k, h, and psi, handling boundaries. """

        h_adjusted = np.clip(h, 1e-6, 1 - 1e-6)  # Avoid zero in power calculations
        m_adjusted = np.clip(m, 1e-6, 1 - 1e-6) # Avoid zero in power calculations
        k_adjusted = max(k, 1e-6)  # Avoid zero in power calculations

        mid_low_input = ( self.theta * (m_adjusted ** self.eta) + (1-self.theta) * ((1-h_adjusted-m_adjusted) ** self.eta) ) ** (1/self.eta)

        composite_input = (self.A_H * h_adjusted)**psi + (self.A_L * (mid_low_input) )**psi
        
        Y = k_adjusted**self.alpha * composite_input**((1 - self.alpha) / psi) 


        k_next = self.s_k * Y + (1 - self.delta) * k_adjusted - k_adjusted
        h_next = self.s_h * Y + (1 - self.delta) * h_adjusted - h_adjusted
        m_next = self.s_m * Y + (1 - self.delta) * m_adjusted - m_adjusted

        return k_next, h_next, m_next

    def find_steady_state_ext(self, k_init, h_init, m_init, psi):
        """ Finds the steady state starting from initial values. """
        solution = optimize.root(lambda vars: self.compute_equilibrium_ext(*vars, psi), [k_init, h_init, m_init], method='hybr')
        if not solution.success:
            print(f"Optimization did not converge: {solution.message}")
        return solution.x