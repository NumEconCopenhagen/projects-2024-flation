# Import packagaes
import numpy as np
np.random.seed(2000)

from scipy import optimize
from scipy.stats import norm

from types import SimpleNamespace
import matplotlib.pyplot as plt
import Examproject
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

class ProductionEconomy :

    def __init__(self) :

        par = self.par = SimpleNamespace()

        # Parameters of the firms
        par.A = 1.0
        par.gamma = 0.5
        
        # Parameters of the households
        par.alpha = 0.3
        par.nu = 1.0
        par.eps = 2.0

        # Government parameters
        par.tau = 0.0
        par.T = 0.0

        # Other parameters
        par.w = 1.0 # Wages are set as numeraire
        par.kappa = 0.1 # Social cost of carbon from the production of good 2  


    def firm_j(self,p) : 
        ''' Computes optimal firm behaviour'''

        par = self.par

        # Optimal firm labor demand 
        ell_j = ( ( p * par.A * par.gamma ) / par.w ) ** ( 1 / ( 1-par.gamma ) )
        
        # Optimal firm output
        y_j = par.A * ( ell_j ) ** par.gamma
        
        # Implied firm profits
        pi_j = ( ( 1 - par.gamma ) / par.gamma ) * par.w * ell_j 
        
        return ell_j, y_j, pi_j
    

    def consumer(self,p1,p2,ell) :

        par = self.par

        _, _, pi_1 = self.firm_j(p1)
        _, _, pi_2 = self.firm_j(p2)

        # Consumption of the good supplied by firm 1
        cons_1 = par.alpha * ( par.w * ell + par.T + pi_1 + pi_2 ) / p1 

        # Consumption of the good supplied by firm 2
        cons_2 = ( 1 - par.alpha ) * ( par.w * ell + par.T + pi_1 + pi_2 ) / (p2 + par.tau)

        # Utility for a given ell
        util = np.log( ( cons_1 ** par.alpha ) * ( cons_2 ** ( 1 - par.alpha ) ) ) - par.nu * ( ell ** ( 1 + par.eps) ) / ( 1 + par.eps )

        return cons_1, cons_2, util
    

    def optimal_consumption(self,p1,p2) :

        # We define the objective function to be minimized wrt. ell
        def objective_func(ell,p1,p2) :
            _, _, util = self.consumer(p1,p2,ell)
            return -util
        
        # We emplyt optimize.minimize_scalar to find the ell that minimizes the objective function
        sol = optimize.minimize_scalar(objective_func,
                                       method = 'bounded',
                                       bounds = (0,2),
                                       args=(p1, p2) )
        ell_star = sol.x
        c1_star, c2_star, _ = self.consumer(p1,p2,ell_star)

        return ell_star, c1_star, c2_star
    

    def check_market_clearing(self,p1,p2) :

        # Household comsumption and labor supply
        ell_star, c1_star, c2_star = self.optimal_consumption(p1,p2)

        # Optimal behaviour of firm 1
        ell_1, y_1, _ = self.firm_j(p1)

        # Optimal behaviour of firm 2
        ell_2, y_2, _ = self.firm_j(p2)

        # Check market clearing conditions
        labor_mkt_clearing = ell_star - ( ell_1 + ell_2 )
        good1_mkt_clearing = c1_star - y_1
        good2_mkt_clearing = c2_star - y_2

        return labor_mkt_clearing, good1_mkt_clearing, good2_mkt_clearing
    

    def compute_equilibrium(self,initial_guess) :

        def objective_func(p) :
            _,  good1_mkt_clearing, good2_mkt_clearing = self.check_market_clearing(p[0],p[1])
            return good1_mkt_clearing, good2_mkt_clearing
        
        x0 = initial_guess

        res = optimize.root(objective_func, x0, method = 'hybr')

        return res.x[0],res.x[1]
    

class Career :

    def __init__(self) :

        par = self.par = SimpleNamespace()

        par.J = 3                       # Career tracks
        par.N = 10                      # Graduates
        par.K = 10000                   # Iterations in simulation

        par.sigma = 2                   # Standard deviations for noise terms
        par.v = np.array([1,2,3])       # Career track utilities

        par.F = np.arange(1,par.N + 1)  # Friends of graduate i = 1, 2, ..., N
        par.c = 1                       # Cost of switching careers


    def expected_career_util(self) :

        par = self.par

        # Create 1 x J numpy arrays to hold the expected utility and average realized utility for each career track
        exp_util = np.zeros(par.J)
        avg_real_util = np.zeros(par.J)

        # For each career path noise terms are drawn and expected utilit and average realized utility is computed
        for j in range(par.J) :

            # Draw the 1 x K array of noise terms randomly from the normal distribution with mean 0 and standard deviation 2
            noise_terms = np.random.normal(0, par.sigma, par.K)

            # Compute expected utility as the sum v_j and the average of the K noise terms
            exp_util[j] = par.v[j] + np.mean(noise_terms)

            # Compute the average realized utility as the average of the sum of v_j and the noise term k
            avg_real_util[j] = np.mean(par.v[j] + noise_terms)

        return exp_util, avg_real_util
    

    def career_choice(self, i, allow_change = False) :

        par = self.par

        # 1 x 3 arrays to hold the prior expected and posterior realized utilities of graduate i
        prior = np.zeros(par.J)
        post = np.zeros(par.J)

        # For each career path of graduate i their i friends noise terms are drawn and the prior expected utility is computed
        for j in range(par.J) :

            noise_term_f = np.random.normal(0, par.sigma, i)
            prior[j] = np.mean(par.v[j] + noise_term_f)

        # The J noise terms of graduate i are drawn and their posterior realized utilities are calculated 
        noise_term_i = np.random.normal(0, par.sigma, par.J)
        post = par.v + noise_term_i

        # Based on the prior expected utility graduate i choose their career track. The prior expected utilit and posterior realized utility of the career track are saved.
        track = np.argmax(prior) + 1
        track_prior = np.max(prior)
        track_post = post[track - 1]

        # Question 2: We don't allow for graduate i to switch careers.  
        if allow_change == False :
            return track, track_prior, track_post
        
        # Question 3: We allow for graduate i to switch careers.
        else :
            
            # We save the previous track
            prev_track = track

            # The prior expected utility and posterior expected utility are corrected for the career switching cost
            for j in range(par.J) :

                if j + 1 == prev_track :
                    continue
                else:
                    prior[j] = prior[j] - par.c
                    post[j] = post[j] - par.c

            # Based on the posterior realized utilities corrected for the job switching cost the career track after year 1 for graduate i is chosen.
            new_track = np.argmax(post) + 1
            new_track_prior = prior[new_track - 1]
            new_track_post = np.max(post)

            # Compute if an indicator to show if graduate i changed career
            if new_track != track :
                change = 1
            else :
                change = 0

            return prev_track, new_track, new_track_prior, new_track_post, change


    def simulate_career(self, i , allow_change = False) :

        par = self.par

        # Allocate arrays for career choice of graduate i for all K simulations
        sim_track = np.zeros(par.K)
        sim_prior = np.zeros(par.K)
        sim_post = np.zeros(par.K)
        sim_prev_track = np.zeros(par.K) # Only used if allow_change = True
        change_each_track = np.zeros(par.J) # Only used if allow_change = True

        for k in range(par.K) :
            
            # Compute the career choice of graduate i for iteration k

            if allow_change == False :

                track, track_prior, track_post = self.career_choice(i)

            else :

                # Compute the original and the new track together with the new prior, new posterior and career change indicator
                prev_track,track, track_prior, track_post, change = self.career_choice(i, allow_change = True)

                # Save the career choice before change was allowed
                sim_prev_track[k] = prev_track

                # Compute how many from each career track switches careers
                for j in range(par.J) :
                    if prev_track == j + 1 and change == 1 :
                        change_each_track[j] = change_each_track[j] + 1

            sim_track[k] = track
            sim_prior[k] = track_prior
            sim_post[k] = track_post

        if allow_change == False :

            return sim_track, sim_prior, sim_post

        else :

            return sim_prev_track, sim_track, sim_prior, sim_post, change_each_track
        

    def compute_results(self, simulation, allow_change = False) :

        par = self.par

        if allow_change == False :

            sim_track, sim_prior, sim_post = simulation

            # Allocate arrays for the share that chooses each of the J career paths
            track_share = np.zeros(par.J)

            for j in range(par.J) :
                track_indicator = np.array( [ [1 if sim_track[m] == j + 1 else 0 ] for m in range(par.K) ]).T[0]
                track_share[j] = np.mean(track_indicator)

        else :
            
            sim_prev_track, sim_track, sim_prior, sim_post, change_each_track = simulation

            # Allocate array to hold the total number if graduate i's that chose each track
            prev_total_each_track = np.zeros(par.J) 

            for j in range(par.J) :
                prev_total_each_track[j] = np.count_nonzero(sim_prev_track == j + 1, axis = 0)

            # Compute the share that switches from each track
            change_share = change_each_track / prev_total_each_track 

        # Calculate the average prior expected utility and posterior realized utility for graduate i
        # These arrays will compute the values of the original choices if allow_change = False. If allow_change == True then they will hold the updated prior and posteriors.
        avg_prior = np.mean(sim_prior)
        avg_post = np.mean(sim_post)

        if allow_change == False :

            return track_share, avg_prior, avg_post
        
        else :

            return change_share, avg_prior, avg_post


    def plot_results(self,results) :

        par = self.par

        track_share, avg_prior, avg_post = results

        fig, ax = plt.subplots(3,1,figsize = (10,20) )

        # Graph 1: Stacked bar chart to show the share of simulations where graduate i chose 1 of the J career tracks
        ax[0].bar(par.F, track_share[:,0], label = 'Track 1', color = 'midnightblue') 
        ax[0].bar(par.F, track_share[:,1], bottom = track_share[:,0], label = 'Track $', color = 'cornflowerblue') 
        ax[0].bar(par.F, track_share[:,2], bottom = track_share[:,0] + track_share[:,1], label = 'Track 3', color = 'lightsteelblue')

        ax[0].set_title('Share of each of the N graduates choosing each career', fontsize = 17)
        ax[0].set_xlabel('Graduate')
        ax[0].set_ylabel('Share choosing a given career')
        ax[0].set_xticks(par.F)
        ax[0].legend()

        # Graph 2: Bar chart to show the average prior expected utility for each graduate
        ax[1].bar(par.F, avg_prior, color = 'cornflowerblue')
        
        ax[1].set_title('Average prior expected utility for each of the N graduates', fontsize = 17)
        ax[1].set_xlabel('Graduate')
        ax[1].set_ylabel('Average prior expected utility')
        ax[1].set_xticks(par.F)

        # Graph 3: Bar chart to show the average posterior realized utility for each graduate
        ax[2].bar(par.F, avg_post, color = 'cornflowerblue')

        ax[2].set_title('Average posterior realized utility for each of the N graduates', fontsize = 17)
        ax[2].set_xlabel('Graduate')
        ax[2].set_ylabel('Average posterior realized utility')
        ax[2].set_xticks(par.F)

        fig.tight_layout(pad=2.5)


    



        













