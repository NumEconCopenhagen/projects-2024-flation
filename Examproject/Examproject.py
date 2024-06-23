# Import packagaes
import numpy as np
np.random.seed(2000)

from scipy import optimize
from scipy.stats import norm

from types import SimpleNamespace
import matplotlib.pyplot as plt
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

        print('The class ProductionEconomy has been initialized')


    def firm_j(self,p) : 

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
    

    def plot_market_clearing(self,p1_array,p2_array,conditions) :
        
        labor_mkt_clearing, good1_mkt_clearing, good2_mkt_clearing = conditions
        # X, Y = np.meshgrid(p2_array , labor_mkt_clearing)
        # two_dim = 

        fig, ax = plt.subplots(3, 1, figsize = (9,27)) 

        # Graph 1: Labor market clearing condition
        for i,j in enumerate(p1_array) :
            p1 = round(j,2)
            ax[0].scatter(p2_array,labor_mkt_clearing[i,:], label = p1)

        ax[0].set_title('Labor market clearing condition for different $p_1$') 

        # Graph 2: Good market 1 clearing condition
        for i,j in enumerate(p1_array) :
            p1 = round(j,2)
            ax[1].scatter(p2_array,good1_mkt_clearing[i,:], label = p1)
   
        ax[1].set_title('Good market 1 clearing condition for different $p_1$')

        # Graph 3: Good market 2 clearing condition
        for i,j in enumerate(p1_array) :
            p1 = round(j,2)
            ax[2].scatter(p2_array,good2_mkt_clearing[i,:], label = p1)

        ax[2].set_title('Good market 2 clearing condition for different $p_1$')  
        
        for i in range(2) :
            ax[i].set_xlabel('$p_2$')
            ax[i].set_xticks(np.round(p2_array,2))
            ax[i].set_ylabel('Value of market clearing condition')
            ax[i].legend(frameon=True,loc='upper right',bbox_to_anchor=(1.2,1.0), title = '$p_1$ values') 


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

        print('The class Career has been initialized')


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


    def plot_results(self,results_q2,results_q3) :

        par = self.par

        track_share, avg_prior, avg_post = results_q2
        change_share, new_avg_prior, new_avg_post = results_q3

        fig, ax = plt.subplots(3,2,figsize = (20,20) )

        # Graphs to visualize the results of question 2

        # Graph 1: Stacked bar chart to show the share of simulations where graduate i chose 1 of the J career tracks
        ax[0,0].bar(par.F, track_share[:,0], label = 'Track 1', color = 'midnightblue') 
        ax[0,0].bar(par.F, track_share[:,1], bottom = track_share[:,0], label = 'Track $', color = 'cornflowerblue') 
        ax[0,0].bar(par.F, track_share[:,2], bottom = track_share[:,0] + track_share[:,1], label = 'Track 3', color = 'lightsteelblue')

        ax[0,0].set_title('Share of each of the N graduates choosing each career', fontsize = 17)
        ax[0,0].set_xlabel('Graduate')
        ax[0,0].set_ylabel('Share choosing a given career')
        ax[0,0].set_xticks(par.F)
        ax[0,0].legend()

        # Graph 2: Bar chart to show the average prior expected utility for each graduate
        ax[1,0].bar(par.F, avg_prior, color = 'cornflowerblue')
        
        ax[1,0].set_title('Average prior expected utility for each of the N graduates', fontsize = 17)
        ax[1,0].set_xlabel('Graduate')
        ax[1,0].set_ylabel('Average prior expected utility')
        ax[1,0].set_xticks(par.F)

        # Graph 3: Bar chart to show the average posterior realized utility for each graduate
        ax[2,0].bar(par.F, avg_post, color = 'cornflowerblue')

        ax[2,0].set_title('Average posterior realized utility for each of the N graduates', fontsize = 17)
        ax[2,0].set_xlabel('Graduate')
        ax[2,0].set_ylabel('Average posterior realized utility')
        ax[2,0].set_xticks(par.F)

        # Graphs to visualize the results of question 3

        # Number of graduates is par.N
        # Position on the x-axis is par.F

        width = 0.2

        ax[0,1].bar(par.F - width, change_share[:,0], width, color = 'maroon', label = 'Track 1')
        ax[0,1].bar(par.F, change_share[:,1], width, color = 'indianred', label = 'Track 2')
        ax[0,1].bar(par.F + width, change_share[:,2], width, color = 'lightcoral', label = 'Track 3')

        ax[0,1].set_title('Share of each of the N graduates switching careers given initial track', fontsize = 17)
        ax[0,1].set_xlabel('Graduate')
        ax[0,1].set_ylabel('Share who chooses to switch careers')
        ax[0,1].set_xticks(par.F)
        ax[0,1].legend()

        # Graph 5: Bar chart to show the average prior expected utility for each graduate
        ax[1,1].bar(par.F, new_avg_prior, color = 'indianred')
        
        ax[1,1].set_title('Average prior expected utility for each of the N graduates', fontsize = 17)
        ax[1,1].set_xlabel('Graduate')
        ax[1,1].set_ylabel('Average prior expected utility')
        ax[1,1].set_xticks(par.F)

        # Graph 6: Bar chart to show the average posterior realized utility for each graduate
        ax[2,1].bar(par.F, new_avg_post, color = 'indianred')

        ax[2,1].set_title('Average posterior realized utility for each of the N graduates', fontsize = 17)
        ax[2,1].set_xlabel('Graduate')
        ax[2,1].set_ylabel('Average posterior realized utility')
        ax[2,1].set_xticks(par.F)

        fig.tight_layout(pad=2.5)


class BarycentricInterpolation :

    def __init__(self) :

        par = self.par = SimpleNamespace()

        par.points_to_find = 4

        print('The class BarycentricInterpolation has been initialized')


    def objective_func(self,x1,x2,y1,y2) :
        return ( (x1 - y1) ** 2 + (x2 - y2) ** 2 ) ** (1/2) 
    

    def constraints(self,x1,x2,y1,y2) :
        return np.array([x1 > y1 and x2 > y2,x1 > y1 and x2 < y2,x1 < y1 and x2 < y2,x1 < y1 and x2 > y2])


    def compute_points(self,X,y) :

        par = self.par

        points = np.zeros([4,2])
        y1, y2 = y

        for point, cons in zip(points, range(par.points_to_find)) :

            value = np.inf

            for x1,x2 in X :

                # Calculate all 4 constraints
                constraint = self.constraints(x1,x2,y1,y2)

                # Evaluate the relevant constraint and return the function value if the constraint is true
                if constraint[cons] == True :
                    temp_val = self.objective_func(x1,x2,y1,y2)

                # If the constraint is not true return NaN
                else :
                    temp_val = np.NaN

                # If the calculated function value is the smallest yet update the variable value and the point
                if temp_val < value :
                    value = temp_val
                    point = [x1,x2]

                points[cons] = point
            
        # Define points
        A = points[0]
        B = points[1]
        C = points[2]
        D = points[3]

        return A,B,C,D
    

    def plot_bary(self,A,B,C,D,X,y) :
        fig, ax = plt.subplots(1,1, figsize = (10,10))

        ax.scatter(X[:,0],X[:,1], label = 'X')
        ax.scatter(A[0],A[1],marker = 's', label = 'A')
        ax.scatter(B[0],B[1],marker = 's', label = 'B')
        ax.scatter(C[0],C[1],marker = 's', label = 'C')
        ax.scatter(D[0],D[1],marker = 's', label = 'D')
        ax.scatter(y[0],y[1],marker = 's', label = 'y')

        # if A is not None and B is not None and C is not None :
        ax.plot( [ A[0], B[0] ] , [ A[1], B[1] ], color = 'cyan', label = 'ABC' )
        ax.plot( [ B[0], C[0] ] , [ B[1], C[1] ], color = 'cyan' )
        ax.plot( [ A[0], C[0] ] , [ A[1], C[1] ], color = 'cyan' )

        ax.plot( [ C[0], D[0] ] , [ C[1], D[1] ], color = 'red', label = 'CDA' )
        ax.plot( [ D[0], A[0] ] , [ D[1], A[1] ], color = 'red' )
        ax.plot( [ A[0], C[0] ] , [ A[1], C[1] ], color = 'red' )

        ax.legend()


    def compute_bary_coordinates(self,P1,P2,P3,y) :

        y1, y2 = y

        r1_num = (P2[1] - P3[1]) * (y1 - P3[0]) + (P3[0] - P2[0]) * (y2 - P3[1])
        r1_denum = (P2[1] - P3[1]) * (P1[0] - P3[0]) + (P3[0] - P2[0]) * (P1[1] - P3[1])

        r2_num = (P3[1] - P1[1]) * (y1 - P3[0]) + (P1[0] - P3[0]) * (y2 - P3[1])
        r2_denum = (P2[1] - P3[1]) * (P1[0] - P3[0]) + (P3[0] - P2[0]) * (P1[1] - P3[1])

        return r1_num / r1_denum, r2_num / r2_denum, 1 - r1_num / r1_denum - r2_num / r2_denum

    def algorithm(self,X,y,f) :
        
        # 1. Compute A, B, C and D. If not possible return NaN.
        A,B,C,D = self.compute_points(X,y)

        # 2. If y is inside the triangle ABC return ...
        if all(0 <= i <= 1 for i in self.compute_bary_coordinates(A,B,C,y)) :
            f_approx = np.sum( np.array(self.compute_bary_coordinates(A,B,C,y)) * np.array([f(A),f(B),f(C)]) )
    
        # 3. If y is inside the triangle CDA return ...
        elif all(0 <= i <= 1 for i in self.compute_bary_coordinates(C,D,A,y)) :
            f_approx = np.sum( np.array(self.compute_bary_coordinates(C,D,A,y)) * np.array([f(C),f(D),f(A)]) )
            
        # 4. Return NaN.
        else :
            f_approx = np.NaN

        return f_approx
        


    



        













