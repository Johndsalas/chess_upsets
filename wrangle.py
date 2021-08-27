'''Aquire and prep chess game data'''

import os
import pandas as pd

def wrangle_chess_data():

    if os.path.isfile('chess_games_prepared.csv') == False:

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

        # binning qualitative data as unpopuler
        for column in columns:
        
            value_set = set(df[column].to_list())
        
            value_set_above_50 = [value for value in value_set if df[column].value_counts()[value] >= 50]

            df[column] = df[column].apply(lambda value : value if value in value_set_above_50 else 'Unpopuler')

        df.to_csv('chess_games_prepared.csv', index = False)

    return pd.read_csv('chess_games_prepared.csv')





