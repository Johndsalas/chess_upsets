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

        # get feature columns
        df = df[['rated','winner', 'increment_code', 'white_rating',
            'black_rating', 'opening_name']]

        # rename columns
        df = df.rename(columns={'winner': 'winning_pieces'})

        # ensuring no white space in values
        columns = ['winning_pieces','increment_code', 'opening_name','opening_name']

        for column in columns:
        
            df[column] = df[column].apply(lambda value: value.strip())

        # adding derived features
        df = add_features(df)

        df = df[['time_control_group','opening_name']].join(pd.get_dummies(df,columns=['time_control_group','opening_name']))

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
    value = int(value[0])

    # return time block
    if value >= 60:

        return "Standard"

    elif value <= 30 and value >= 15:

        return "Rapid"

    elif value <= 5 and value >= 3:

        return "Blitz"

    elif value <= 2:

        return "Bullet"

    else:

        return "Other"

def add_features(df):
    '''Adds features for exploration'''

    df['upset'] = (((df.white_rating > df.black_rating) & (df.winning_pieces == 'black')) |
                  ((df.white_rating < df.black_rating) & (df.winning_pieces == 'white')))

    df["rating_difference"] = abs(df.white_rating - df.black_rating)

    df["game_rating"] = (df.white_rating + df.black_rating) / 2
    df["game_rating"] = df["game_rating"].astype(int)

    df["lower_rated_white"] = (df.white_rating < df.black_rating)

    df['time_control_group'] = df.increment_code.apply(lambda value: get_time_block(value))

    return df

#######################Holding off on post split questions for now***********************************

#*******************************************Post Split***********************************************

# def get_time_control(value):
#     '''convert time code to time in minutes'''

#     # get both variables from the time code
#     value = re.sub(r'\+', ' ', value)
#     value = value.split(' ')

#     # return overall time-control
#     return value[1]

# def fe_post_split(train, validate, test):
#     '''Adds features to data post splitting'''

#     # add opening_ave_rating column (average rating of games that are played using this opening)
#     train["opening_ave_rating"] = train.opening_name.apply(lambda value : train[train.opening_name == value].game_rating.mean())
#     validate["opening_ave_rating"] = train.opening_name.apply(lambda value : train[train.opening_name == value].game_rating.mean())
#     test["opening_ave_rating"] = train.opening_name.apply(lambda value : train[train.opening_name == value].game_rating.mean())

#     # add opening_popularity (total games played with this opening in train)
#     train['opening_popularity_total'] = train['opening_name'].apply(lambda value : len(train[train.opening_name == value]))
#     validate['opening_popularity'] = validate['opening_name'].apply(lambda value : len(train[train.opening_name == value]))
#     test['opening_popularity'] = test['opening_name'].apply(lambda value : len(train[train.opening_name == value]))
    
#     # add opening_popularity_1500 (total games played with this opening by players who's average rating is over 1500 in train)
#     train['opening_popularity_1500'] = train['opening_name'].apply(lambda value : len(train[(train.game_rating > 1500) & (train.opening_name == value)]))
#     validate['opening_popularity_1500'] = validate['opening_name'].apply(lambda value : len(train[(train.game_rating > 1500) & (train.opening_name == value)]))
#     test['opening_popularity_1500'] = test['opening_name'].apply(lambda value : len(train[(train.game_rating > 1500) & (train.opening_name == value)]))

#     # add opening_popularity_2000 (total games played with this opening by players who's average rating is over 2000 in train)
#     train['opening_popularity_2000'] = train['opening_name'].apply(lambda value : len(train[(train.game_rating > 2000) & (train.opening_name == value)]))
#     validate['opening_popularity_2000'] = validate['opening_name'].apply(lambda value : len(train[(train.game_rating > 2000) & (train.opening_name == value)]))
#     test['opening_popularity_2000'] = test['opening_name'].apply(lambda value : len(train[(train.game_rating > 2000) & (train.opening_name == value)]))

#     return train, validate, test