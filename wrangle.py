'''Aquire and prep chess game data'''

import os
import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

####################################Acquire and Prep##################################################

def wrangle_chess_data(reprep = False):
    ''' Aquires and Prepares data for project'''

    if (os.path.isfile('chess_games_prepared.csv') == False) or (reprep == True):

        # read in data from csv
        df = pd.read_csv('games.csv')

        # get target columns
        df = df[['rated', 'turns', 'victory_status',
            'winner', 'increment_code', 'white_rating',
            'black_rating', 'opening_eco', 'opening_name']]

        # rename columns
        df = df.rename(columns={'victory_status':'ended_as',
                            'increment_code':'time_code',
                            'opening_eco':'opening_code',
                            'winner': 'winning_pieces'})

        # ensuring no white space in values
        columns = ['ended_as', 'winning_pieces',
                   'time_code', 'opening_name',
                   'opening_name']

        for column in columns:
        
            df[column] = df[column].apply(lambda value: value.strip())

        # adding pre-split features
        df = fe_pre_split(df)

        # saving to csv
        df.to_csv('chess_games_prepared.csv', index = False)

    return pd.read_csv('chess_games_prepared.csv')

####################################Trian Validate Test Split########################################

def split_my_data(df):
    '''Splits data into train, validate, and test data'''

    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.upset)

    train, validate =  train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.upset)

    return train, validate, test

########################################Feature Engineering##############################################

#*******************************************Pre Split***********************************************

def get_time_block(value):
    '''convert time code to time in minutes'''

    # get both variables from the time code
    value = re.sub(r'\+', ' ', value)
    value = value.split(' ')

    # return time block
    return value[0]


def fe_pre_split(df):
    '''Adds features to data before splitting'''

    df['upset'] = (((df.white_rating > df.black_rating) & (df.winning_pieces == 'black')) |
                  ((df.white_rating < df.black_rating) & (df.winning_pieces == 'white')))

    df["rating_dif"] = abs(df.white_rating - df.black_rating)

    df["game_rating"] = (df.white_rating + df.black_rating) / 2
    df["game_rating"] = df["game_rating"].astype(int)

    df["lower_rated_white"] = (df.white_rating < df.black_rating)

    df["time_block"] = df.time_code.apply(lambda value: get_time_block(value))

    return df

#*******************************************Post Split***********************************************

def get_time_in_minutes(value,average_moves):
    '''convert time code to time in minutes'''

    # get both variables from the time code
    value = re.sub(r'\+', ' ', value)
    value = value.split(' ')

    # retunr calculated assigned play time for each player assuming average number of moves
    return ((int(value[0]) * 60) + (int(value[1]) * (average_moves/2)))/60

def fe_post_split(train, validate, test):
    '''Adds features to data post splitting'''

    # get averages game based on train data
    average_moves = train.turns.mean()
    average_rating = train.game_rating.mean()

    # add time_minutes column to train validate and test calculation is based on time increment and assumes an average number of moves
    train["time_minutes"] = train.time_code.apply(lambda value: get_time_in_minutes(value,average_moves))
    validate["time_minutes"] = validate.time_code.apply(lambda value: get_time_in_minutes(value,average_moves))
    test["time_minutes"] = test.time_code.apply(lambda value: get_time_in_minutes(value,average_moves))

    # add opening_code_pop column calculation is based on the percent of games played with this opening in training data
    train['opening_code_pop'] = train['opening_code'].apply(lambda value : len(train[train.opening_code == value])/len(train))
    validate['opening_code_pop'] = validate['opening_code'].apply(lambda value : len(train[train.opening_code == value])/len(train))
    test['opening_code_pop'] = test['opening_code'].apply(lambda value : len(train[train.opening_code == value])/len(train))
    
    # add opening_name_pop column claculation is based on the percent of games played with this opening name in test data
    train['opening_name_pop'] = train['opening_name'].apply(lambda value : len(train[train.opening_code == value])/len(train))
    validate['opening_name_pop'] = validate['opening_name'].apply(lambda value : len(train[train.opening_code == value])/len(train))
    test['opening_name_pop'] = test['opening_name'].apply(lambda value : len(train[train.opening_code == value])/len(train))

    # add opening_code_high_pop column calculation is based on the percent of games played with this opening amoung above average rated games in the training data
    train['opening_code_high_pop'] = train['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    validate['opening_code_high_pop'] = validate['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    test['opening_code_high_pop'] = test['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    
    # add opening_name_high_pop column claculation is based on the percent of games played with this opening amoung above average rated games in the training data
    train['opening_name_high_pop'] = train['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    validate['opening_name_high_pop'] = validate['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    test['opening_name_high_pop'] = test['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > average_rating)])/len(train))
    
    # add opening_code_top_pop column calculation is based on the percent of games played with this opening amoung games rated +2000 in the training data
    train['opening_code_top_pop'] = train['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))
    validate['opening_code_top_pop'] = validate['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))
    test['opening_code_top_pop'] = test['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))
    
    # add opening_name_top_pop column claculation is based on the percent of games played with this opening amoung games rated +2000 in the training data
    train['opening_name_top_pop'] = train['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))
    validate['opening_name_top_pop'] = validate['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))
    test['opening_name_top_pop'] = test['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 2000)])/len(train))

    # add opening_code_low_pop column calculation is based on the percent of games played with this opening amoung games rated -1000 in the training data
    train['opening_code_low_pop'] = train['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating <= 1000)])/len(train))
    validate['opening_code_low_pop'] = validate['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 1000)])/len(train))
    test['opening_code_low_pop'] = test['opening_code'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 1000)])/len(train))
    
    # add opening_name_low_pop column claculation is based on the percent of games played with this opening amoung games rated -1000 in the training data
    train['opening_name_low_pop'] = train['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 1000)])/len(train))
    validate['opening_name_low_pop'] = validate['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 1000)])/len(train))
    test['opening_name_low_pop'] = test['opening_name'].apply(lambda value : len(train[(train.opening_code == value) & (train.game_rating > 1000)])/len(train))

    # add opening_code_rating_mean column calculation is based on the average game rating of games played using this opening in the training data
    train['opening_code_rating_mean'] = train['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.mean())
    validate['opening_code_rating_mean'] = validate['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.mean())
    test['opening_code_rating_mean'] = test['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.mean())

    # add opening_code_rating_median column calculation is based on the median game rating of games played using this opening in the training data
    train['opening_code_rating_median'] = train['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.median())
    validate['opening_code_rating_median'] = validate['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.median())
    test['opening_code_rating_median'] = test['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.median())

    # add opening_code_rating_max column calculation is based on the max game rating of games played using this opening in the training data
    train['opening_code_rating_max'] = train['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.max())
    validate['opening_code_rating_max'] = validate['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.max())
    test['opening_code_rating_max'] = test['opening_code'].apply(lambda value : train[train.opening_code == value].game_rating.max())

    return train, validate, test