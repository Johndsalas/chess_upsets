import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def get_pie_upsets(train):
    '''get pie chart for percent of upsets'''

    # set values and labels for chart
    values = [len(train.upset[train.upset == True]), len(train.upset[train.upset == False])] 
    labels = ['Upset','Non-Upset', ] 

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    plt.title('Games Ending in Upsets Represent 1/3 of the train data')
    plt.show()

def get_pies_white(train):
    "create pie charts showing upset percentage for having and not having the first move"

    # create axis object
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    
    # create pie chart and assign to axis object
    values = [len(train.upset[(train.lower_rated_white == True) & (train.upset == True)]),
            len(train.upset[(train.lower_rated_white == True) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset']

    ax1.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    ax1.title.set_text('Lower Rated Player has First Move')

    # create pie chart and and assign to axis object
    values = [len(train.upset[(train.lower_rated_white == False) & (train.upset == True)]),
            len(train.upset[(train.lower_rated_white == False) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset'] 

    ax2.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    ax2.title.set_text('Higher Rated Player has First Move')

    # display charts
    plt.tight_layout()
    plt.show()

def get_chi_white(train):
    '''get rusults of chi-square for lower_rated_white and upsets'''

    observed = pd.crosstab(train.lower_rated_white, train.upset)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_pie_rated(train):
    '''get pie charts showing the percentage of upsets for games being rated on not being rated'''
    
    # create axes objects
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))

    # generate pie chart and assign it to ax1
    values = [len(train.upset[(train.rated == True) & (train.upset == True)]),
            len(train.upset[(train.rated == True) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset']

    ax1.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    ax1.title.set_text('Game is Rated')

    # generate pie chart and assign it to ax2
    values = [len(train.upset[(train.rated == False) & (train.upset == True)]),
            len(train.upset[(train.rated == False) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset'] 

    ax2.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    ax2.title.set_text('Game is not Rated')

    # display plot
    plt.tight_layout
    plt.show()

def get_chi_rated(train):
    '''get result of chi-square for rated and upset'''

    observed = pd.crosstab(train.rated, train.upset)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_game_rating(train):
    " get graph of game rating for upsets and non-upsets"

    # assign values and labels
    values = [train.game_rating[(train.upset == True)].mean(),train.game_rating[(train.upset == False)].mean()]
    labels = ['Upset','Non-Upset']

    # generate and display graph
    plt.bar(height=values, x=labels, color=['#ffc3a0', '#c0d6e4'])
    plt.title('The Mean Game Rating is About the Same in Upsets and Non-upsets')
    plt.tight_layout()
    plt.show()

def ave_diff_rating(train):
    " get graph of game mean rating_difference for upsets and non-upsets"

    # assign values and labels
    values = [train.rating_difference[(train.upset == True)].mean(),train.rating_difference[(train.upset == False)].mean()]
    labels = ['Upset','Non-Upset', ]

    # generate and display chart
    plt.bar(height=values, x=labels, color=['#ffc3a0', '#c0d6e4'])
    plt.title('The Mean Difference in Player Rating is Much Smaller in Upsets than in Non-upsets')
    plt.tight_layout()
    plt.show()

def get_t_rating_diff(train):
    "get t-test for mean rating difference in upsets and non-upsets"

    t, p = stats.ttest_ind(train.rating_difference[(train.upset == True)],train.rating_difference[(train.upset == False)])

    print(f't = {t:.4f}')
    print(f'p = {p:.4f}')    

def get_pie_time(train):
    '''get pie chart of upset percentage for each time control'''

    # activate subplots objects
    fig, axs = plt.subplots(2, 2, figsize=(10,8))

    # list of charts to be generated
    times = ['Bullet', 'Blitz', 'Rapid', 'Standard']

    # generate graphs and assign them to subplots
    for time, ax in zip(times, axs.ravel()):
        
        values = [len(train.upset[(train.upset == True) & (train.time_control_group == time)]), len(train.upset[(train.upset == False) & (train.time_control_group == time)])] 
        labels = ['Upset','Non-Upset']
        
        ax.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
        ax.set_title(f'Upset Percentage for time block {time}')
    
    # display chart
    plt.tight_layout()
    plt.show()

def get_chi_time(train):
    ''' get chi-square for upset and time control group'''

    observed = pd.crosstab(train.time_control_group, train.upset)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_pie_open(train):
    '''get pie chart of upset percentage for the top ten openings in the train date by popularity'''

     # activate subplots objects
    fig, axs = plt.subplots(2, 5, figsize=(20,10))

    # list of charts to be generated
    names = train.opening_name.value_counts().head(10).index.to_list()

    # generate graphs and assign them to subplots
    for name, ax in zip(names, axs.ravel()):
        
        values = [len(train.upset[(train.upset == True) & (train.opening_name == name)]), len(train.upset[(train.upset == False) & (train.opening_name == name)])] 
        labels = ['Upset','Non-Upset'] 
        ax.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
        ax.set_title(f'{name}')
           
    # display chart
    plt.tight_layout()
    plt.show()

def get_chi_open(train):
    '''get chi-square for opening name and upset'''

    observed = pd.crosstab(train.opening_name, train.upset)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')    