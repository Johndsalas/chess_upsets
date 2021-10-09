'''Aquire and prep chess game data'''

import os
import pandas as pd
import numpy as np

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

    train, validate =  train_test_split(train_validate, test_size=.3, random_state=123, stratify=df.upset)

    return train, validate, test

########################################Feature Engineering##############################################

def fe_pre_split(df):
    '''Adds features to data before splitting'''

    df['upset'] = (((df.white_rating > df.black_rating) & (df.winning_pieces == 'black')) |
                  ((df.white_rating < df.black_rating) & (df.winning_pieces == 'white')))

    df["rating_dif"] = abs(df.white_rating - df.black_rating)

    df["game_rating"] = (df.white_rating + df.black_rating) / 2
    df["game_rating"] = df["game_rating"].astype(int)

    df["lower_rated_white"] = (df.white_rating < df.black_rating)

    return df