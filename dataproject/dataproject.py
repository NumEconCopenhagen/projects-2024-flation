# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

# List of all 27 EU countries
eu_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", 
    "Czechia", "Denmark", "Estonia", "Finland", "France", 
    "Germany", "Greece", "Hungary", "Ireland", "Italy", 
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", 
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", 
    "Spain", "Sweden"]

# We define our favorite colors
cbs_blue = (73/255, 103/255, 170/255) ; ku_red = (144/255, 26/255, 30/255)


def european(df) :
    rts_eu = df[df['isoname'].isin(eu_countries)]
    return rts_eu

def data_clean_rts(df) :
    # Step b: Remove returns to schooling for primary, secondary and tertiary education as well the extra index as these are not of interest to us.
    # var_ends = ('pri','sec','ter')
    
    # drop_these = [name for name in df.columns if name.endswith(var_ends)]
    # df.drop(drop_these, axis = 1, inplace = True)
    
    for j in ['coef_','lb_','ub_'] :
        drop_these = [j + x for x in ['yrl_pri','yrl_sec','yrl_ter']]
        df.drop(drop_these, axis = 1, inplace = True)

    del df['index']

    # Step c: Rename variables such that they are easy to understand
    rename_dict = {}
    rename_dict['isoname'] = 'country'
    rename_dict['coef_yrs'] = 'return_to_schooling'
    rename_dict['lb_yrs'] = 'lower_rts'
    rename_dict['ub_yrs'] = 'upper_rts'
    rename_dict['coef_yrs_sex_0'] = 'rts_women'
    rename_dict['lb_yrs_sex_0'] = 'lower_rts_women'
    rename_dict['ub_yrs_sex_0'] = 'upper_rts_women'
    rename_dict['coef_yrs_sex_1'] = 'rts_men'		
    rename_dict['lb_yrs_sex_1'] = 'lower_rts_men'
    rename_dict['ub_yrs_sex_1'] = 'upper_rts_men'
    df.rename(columns = rename_dict, inplace = True)

    # Step d: The units are in decimals, we choose to change units into percentages
    for key,value in rename_dict.items() :
        if value != 'country' :
            df[value] = df[value] * 100
        else :
            continue

    # Step e: The dataset is sorted from smallets return to schooling to largest
    rts_eu_sorted = df.sort_values('return_to_schooling').reset_index()

    return rts_eu_sorted

def data_clean_gexp(df) :
    
    # Step b: Keep variables of interest
    keep_these = ['isoname','iso','year','gdp','gova_educ','npop']
    df = df.loc[:,keep_these]

    # Step c: Create per capita variables (in 1000 dollars)
    for i in ['gdp','gova_educ'] :
        df[i + '_cap_t'] = df[i] / (df['npop'] * 1000)
        del df[i]

    # Step d: Rename isoname to country such that it matches the dataset rts_eu_sorted
    df.rename(columns = {'isoname':'country'}, inplace = True)

    # Step e: We remove all years 2020-2022 as they are after the record of return to schooling in 2019
    J = False
    for j in [2020,2021,2022] :
        J |= df.year == j # Laver en ændring her

    df = df.loc[J == False].reset_index()

    # Step f: Calculate the average government expenditures into schooling per capita from 1980-2019
    df['avg_gov_educ_exp'] = df.groupby('country')['gova_educ_cap_t'].transform('mean')

    # Step g: As the return to schooling is noted in 2019 we only keep the relevant year (i.e. gdp per capita for 2019 and average government spending from 1980 to 2019)
    gexp_eu_2019 = df.groupby('country').last().reset_index()

    # Step h: Create the logarithims of the variables for plotting
    gexp_eu_2019['avg_gov_educ_exp_log'] = np.log(gexp_eu_2019['avg_gov_educ_exp'])
    gexp_eu_2019['gdp_cap_t_log'] = np.log(gexp_eu_2019['gdp_cap_t'])

    # Step i: Remove index before merging
    del gexp_eu_2019['index']

    return gexp_eu_2019

def grand_merge(df1,df2) :
    # Doing a left merge on country and iso (We do it on iso as it is in both datasets)
    gexp_rts_eu = pd.merge(df1, df2, on = ['country','iso'], how = 'left')
    del gexp_rts_eu['index']

    return gexp_rts_eu

def plot_country_comp(df) :

    # We use pandas built in plot function to make a bar chart of our favorite pies
    ax = df.plot(x = 'country', y = 'return_to_schooling', kind = 'bar', 
                        figsize=(10, 5), color = cbs_blue, legend = False) ; 

    # We higlight the bar for Denmark
    bars = ax.patches
    DK = df.index[df['country'] == 'Denmark'].tolist()[0]
    bars[DK].set_facecolor('red')

    # We add labels to the chart
    ax.set_xlabel('') 
    ax.set_ylabel('Return to an additional year of schooling (%)', fontsize = 12) ;

def plot_women_men(df) :
    # Make the axes for the 45 degree line
    deg_45_line = np.linspace(6,24,19)

    # Define a figure
    fig = plt.figure(figsize = (9,9))
    ax1 = fig.add_subplot(1,1,1)

    # Plot the return to schooling for women against the one for men as well as the 45 degree line
    ax1.plot(deg_45_line,deg_45_line)
    ax1.scatter(df['rts_men'],df['rts_women'])

    # Give each point a name corresponding to their isocode
    for i, label in enumerate(df['iso']) :
        ax1.text(df['rts_men'][i],df['rts_women'][i],label,fontsize = 9, ha = 'left', va = 'top')

    # Set limits and labels
    ax1.set_ylim(6,24) ; ax1.set_xlim(6,24)
    ax1.set_ylabel('Return to schooling for women (%)')
    ax1.set_xlabel('Return to schooling for men (%)') 
    ax1.set_title('Return to schooling: Women vs. Men') ;

def plot_covariation(df) :
    # We define a figure
    fig = plt.figure(figsize = (17,8))

    # We want two subplots
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # Figure 1: Gov exp and return to schooling
    ax1.scatter(df['avg_gov_educ_exp_log'],df['return_to_schooling'], color = cbs_blue)

    ax1.set_ylabel('Return to schooling (%)')
    ax1.set_xlabel('Log of avg. government expenditures in education (1000$)', fontsize = 14) 
    ax1.set_title('Covariation btw. gov. exp. in and return to schooling')

    # Make a linear fit and plot it
    slope1, intercept1 = np.polyfit(df['avg_gov_educ_exp_log'],df['return_to_schooling'],1)
    yfit1 = slope1 * df['avg_gov_educ_exp_log'] + intercept1
    ax1.plot(df['avg_gov_educ_exp_log'],yfit1, color = cbs_blue, label = 'linear fit')

    ax1.legend()

    # Figure 2: GDP and return to schooling
    ax2.scatter(df['gdp_cap_t_log'], df['return_to_schooling'], color = ku_red)
    # ax2.set_ylabel('Return to schooling (%)',fontsize = 11)
    ax2.set_xlabel('GDP per capita (1000$)',fontsize = 14) 
    ax2.set_title('Covariation btw. GDP per capita and return to schooling')

    # Make a linear fit and plot it
    slope2, intercept2 = np.polyfit(df['gdp_cap_t_log'],df['return_to_schooling'],1)
    yfit2 = slope2 * df['gdp_cap_t_log'] + intercept2
    ax2.plot(df['gdp_cap_t_log'],yfit2, color = ku_red, label = 'Linear fit')

    ax2.legend()

    # Add text labels to all points
    for i, label in enumerate(df['iso']) :
        ax1.text(df['avg_gov_educ_exp_log'][i],df['return_to_schooling'][i],label,fontsize = 9, ha = 'left', va = 'top')
        ax2.text(df['gdp_cap_t_log'][i],df['return_to_schooling'][i],label,fontsize = 9, ha = 'left', va = 'top') ;
