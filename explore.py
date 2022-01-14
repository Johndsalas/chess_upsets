import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def get_pie_upsets(train):

    values = [len(train.upset[train.upset == True]), len(train.upset[train.upset == False])] 
    labels = ['Upset','Non-Upset', ] 
    plt.pie(values, labels=labels, autopct='%.0f%%')
    plt.title('Games Ending in Upsets Represent 1/3 of the train data')
    plt.show()

def get_pies_white(train):

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    #fig.suptitle('Upset Percentage is 4% Higher in Games Where the Lower Rated Player has the First Move')

    values = [len(train.upset[(train.lower_rated_white == True) & (train.upset == True)]),
            len(train.upset[(train.lower_rated_white == True) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset']

    ax1.pie(values, labels=labels, autopct='%.0f%%')
    ax1.title.set_text('Lower Rated Player has First Move')

    values = [len(train.upset[(train.lower_rated_white == False) & (train.upset == True)]),
            len(train.upset[(train.lower_rated_white == False) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset'] 

    ax2.pie(values, labels=labels, autopct='%.0f%%')
    ax2.title.set_text('Higher Rated Player has First Move')

    plt.show()

def get_chi_white(train):

    observed = pd.crosstab(train.lower_rated_white, train.upset)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_pie_rated(train):

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))

    values = [len(train.upset[(train.rated == True) & (train.upset == True)]),
            len(train.upset[(train.rated == True) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset']

    ax1.pie(values, labels=labels, autopct='%.0f%%')
    ax1.title.set_text('Game is Rated')

    values = [len(train.upset[(train.rated == False) & (train.upset == True)]),
            len(train.upset[(train.rated == False) & (train.upset == False)])]
    labels = ['Upset', 'Non-Upset'] 

    ax2.pie(values, labels=labels, autopct='%.0f%%')
    ax2.title.set_text('Game is not Rated')

    plt.show()

def get_chi_rated(train):

    observed = pd.crosstab(train.rated, train.upset)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_game_rating(train):

    values = [train.game_rating[(train.upset == True)].mean(),train.game_rating[(train.upset == False)].mean()]
    labels = ['Upset','Non-Upset', ] 

    plt.bar(height=values, x=labels)
    plt.title('The Mean Game Rating is About the Same in Upsets and Non-upsets')
    plt.show()

def ave_diff_rating(train):

    values = [train.rating_difference[(train.upset == True)].mean(),train.rating_difference[(train.upset == False)].mean()]
    labels = ['Upset','Non-Upset', ] 

    plt.bar(height=values, x=labels)
    plt.title('The Mean Difference in Player Rating is Much Smaller in Upsets than in Non-upsets')
    plt.show()

def get_t_rating_diff(train):

    t, p = stats.ttest_ind(train.rating_difference[(train.upset == True)],train.rating_difference[(train.upset == False)])

    print(f't = {t:.4f}')
    print(f'p = {p:.4f}')    

def get_pie_time(train):

    times = ['Bullet', 'Blitz', 'Rapid', 'Standard']

    for time in times:
        
        values = [len(train.upset[(train.upset == True) & (train.time_control_group == time)]), len(train.upset[(train.upset == False) & (train.time_control_group == time)])] 
        labels = ['Upset','Non-Upset', ] 
        
        plt.pie(values, labels=labels, autopct='%.0f%%')
        plt.title(f'Upset Percentage for time block {time}')
        plt.show()

def get_chi_time(train):

    observed = pd.crosstab(train.time_control_group, train.upset)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def get_pie_open(train):

    names = train.opening_name.value_counts().head(10).index.to_list()

    for name in names:
        
        values = [len(train.upset[(train.upset == True) & (train.opening_name == name)]), len(train.upset[(train.upset == False) & (train.opening_name == name)])] 
        labels = ['Upset','Non-Upset'] 
        plt.pie(values, labels=labels, autopct='%.0f%%')
        plt.title(f'Upset Percentage for {name}')
        plt.show()

def get_chi_open(train):

    observed = pd.crosstab(train.opening_name, train.upset)

    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')    