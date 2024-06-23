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

        # As h is the share of skilled capital we bound it to be between 0 and 1
        h_bounded = np.clip(h, 1e-6, 1 - 1e-6)
        
        # The capital stock is non negative
        k_bounded = max(k, 1e-6)

        # The composite input is the the part of the CES production function in parantheses
        composite_input = (self.A_H * h_bounded)**psi + (self.A_L * (1 - h_bounded))**psi
        
        # Compute output per capita
        y = k_bounded**self.alpha * composite_input**((1 - self.alpha) / psi)  # Corrected use of psi
        
        # Compute capital accumulation per capita
        k_next = self.s_k * y + (1 - self.delta) * k_bounded - k_bounded
        
        # Compute skilled labor accumulation per capita
        h_next = self.s_h * y + (1 - self.delta) * h_bounded - h_bounded
        
        return k_next, h_next


    def find_steady_state(self, k_initial, h_initial, psi):
        
        # We use the root function from scipy to find the steady state as the point k_next and h_next are equal to zero
        solution = optimize.root(lambda i : self.output_and_accumulation(*i, psi), [k_initial, h_initial], method='hybr')
        
        # In case the optimiziation doesn't converge we show an error message
        if not solution.success:
            print(f"Optimization did not converge: {solution.message}")
        
        # Calculate output per capita
        composite_input = (self.A_H * solution.x[1])**psi + (self.A_L * (1 - solution.x[1]))**psi
        y = solution.x[0]**self.alpha * composite_input**((1 - self.alpha) / psi)

        return solution.x[0], solution.x[1], y


    def plot_phase_diagram(self, k_range, h_range, psi):

        # Make k_range and h_range into grids
        K, H = np.meshgrid(k_range, h_range)
        
        # We apply the output_and_accumulation function to the K and H grids to get grids for k_next = 0 and h_next = 0
        results = np.vectorize(lambda k, h: self.output_and_accumulation(k, h, psi),signature='(),()->(),()')(K, H)
        
        DeltaK_0 = results[0]
        DeltaH_0 = results[1]

        # We find the steady state value
        k_ss, h_ss, _ = self.find_steady_state(0.5, 0.3, psi)

        # Make the figure
        fig, ax = plt.subplots(1,1,figsize=(8, 6))

        # To plot the phase diagram we plot two contour plots

        # We first plot nullcline for deltak = 0
        contours_k = ax.contour(K, H, DeltaK_0, levels=[0], colors='blue', linestyles='solid')
        
        # We then plot the nullcline for deltah = 0
        contours_h = ax.contour(K, H, DeltaH_0, levels=[0], colors='red', linestyles='solid')

        # We plot the steady state point
        ax.scatter(k_ss,h_ss, marker = 's', color = 'black')
        
        # We add the labels on the lines because it looks cool
        ax.clabel(contours_k, inline=True, fontsize=8, fmt={0: 'Δk = 0'})
        ax.clabel(contours_h, inline=True, fontsize=8, fmt={0: 'Δh = 0'})

        # We set some options for the plot
        ax.set_title(f'Phase Diagram with ψ = {psi}')
        ax.set_xlabel('Capital per capita (k)')
        ax.set_ylabel('Skilled labor per capita (h)')


    def compute_equilibrium_ext(self, k, h, m, psi):

        # As h and m is the share of high skilled and medium skilled workers we bound it to be between 0 and 1
        h_bounded = np.clip(h, 1e-6, 1 - 1e-6)  
        m_bounded = np.clip(m, 1e-6, 1 - 1e-6) 

        # The capital stock is non negative
        k_bounded = max(k, 1e-6)  

        # We compute output per capita
        mid_low_input = ( self.theta * (m_bounded ** self.eta) + (1-self.theta) * ((1-h_bounded-m_bounded) ** self.eta) ) ** (1/self.eta)
        composite_input = (self.A_H * h_bounded)**psi + (self.A_L * (mid_low_input) )**psi
        y = k_bounded**self.alpha * composite_input**((1 - self.alpha) / psi) 

        # Compute accumulation functions
        k_next = self.s_k * y + (1 - self.delta) * k_bounded - k_bounded
        h_next = self.s_h * y + (1 - self.delta) * h_bounded - h_bounded
        m_next = self.s_m * y + (1 - self.delta) * m_bounded - m_bounded

        return k_next, h_next, m_next


    def find_steady_state_ext(self, k_init, h_init, m_init, psi):
        
        # We use the root function from scipy to find the steady state as the point k_next, m_next and h_next are equal to zero
        solution = optimize.root(lambda i: self.compute_equilibrium_ext(*i, psi), [k_init, h_init, m_init], method='hybr')
        
        # In case the optimiziation doesn't converge we show an error message
        if not solution.success:
            print(f"Optimization did not converge: {solution.message}")

        # Compute output per capita
        mid_low_input = ( self.theta * (solution.x[2] ** self.eta) + (1-self.theta) * ((1-solution.x[1]-solution.x[2]) ** self.eta) ) ** (1/self.eta)
        composite_input = (self.A_H * solution.x[1])**psi + (self.A_L * (mid_low_input) )**psi
        y = solution.x[0]**self.alpha * composite_input**((1 - self.alpha) / psi) 

        return solution.x[0], solution.x[1], solution.x[2], y

        