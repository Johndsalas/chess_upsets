
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


def model_prep(train,validate,test):

    drop_cols = ['time_control_group',
                 'opening_name',
                 'winning_pieces',
                 'increment_code',
                 'white_rating',
                 'black_rating',
                 'game_rating',]

    train = train.drop(columns=drop_cols)
    validate = validate.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)
    
    # Splitting data into predicting variables (X) and target variable (y) and resetting the index for each dataframe
    train_X = train.drop(columns='upset').reset_index(drop=True)
    train_y = train[['upset']].reset_index(drop=True)

    validate_X = validate.drop(columns='upset').reset_index(drop=True)
    validate_y = validate[['upset']].reset_index(drop=True)

    test_X = test.drop(columns='upset').reset_index(drop=True)
    test_y = test[['upset']].reset_index(drop=True)

    # Scaling continuous variables

    # columns to be scaled
    cols_to_scale = ['rating_difference']

    # create df's for train validate and test with only columns that need to be scaled
    train_to_be_scaled = train_X[cols_to_scale]
    validate_to_be_scaled = validate_X[cols_to_scale]
    test_to_be_scaled = test_X[cols_to_scale]

    # create scaler object and fit that object on the train data
    scaler = sklearn.preprocessing.MinMaxScaler().fit(train_to_be_scaled)

    # transform data into an array using the scaler object 
    train_scaled = scaler.transform(train_to_be_scaled)
    validate_scaled = scaler.transform(validate_to_be_scaled)
    test_scaled = scaler.transform(test_to_be_scaled)

    # transform data into a dataframe
    train_scaled = pd.DataFrame(train_scaled, columns = cols_to_scale)
    validate_scaled = pd.DataFrame(validate_scaled, columns = cols_to_scale)
    test_scaled = pd.DataFrame(test_scaled, columns = cols_to_scale)

    # drop corresponding unscaled columns from original dataframes and reset the index
    train_X = train_X.drop(columns = cols_to_scale)
    validate_X = validate_X.drop(columns = cols_to_scale)
    test_X = test_X.drop(columns = cols_to_scale)

    # add scaled columns to their original dataframes
    train_X = train_X.join(train_scaled)
    validate_X = validate_X.join(validate_scaled)
    test_X = test_X.join(test_scaled)

    train_X['rated'] = train_X.rated.apply(lambda value: 1 if value == True else 0)
    train_X['lower_rated_white'] = train_X.lower_rated_white.apply(lambda value: 1 if value == True else 0)

    # Change target column to show values as upset or non-upset
    train_y['upset'] = train_y.upset.apply(lambda value: "upset" if value == True else "non-upset")
    validate_y['upset'] = validate_y.upset.apply(lambda value: "upset" if value == True else "non-upset")
    test_y['upset'] = test_y.upset.apply(lambda value: "upset" if value == True else "non-upset")

    return train_X, validate_X, test_X, train_y, validate_y, test_y

def get_tree(train_X, validate_X, train_y, validate_y):

    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5, random_state=123)

    #fit model on training data
    clf = clf.fit(train_X, train_y)

    # get model accuracy for Decision Tree
    print(f"Accuracy of Decision Tree on train data is {clf.score(train_X, train_y)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(validate_X, validate_y)}")

def get_forest(train_X, validate_X, train_y, validate_y):

    rf = RandomForestClassifier(max_depth=9, random_state=123)

    rf.fit(train_X,train_y)

    # get model accuracy for training data
    print(f"Accuracy of Random Forest on train is {rf.score(train_X, train_y)}")
    print(f"Accuracy of Random Forest on validate is {rf.score(validate_X, validate_y)}")

def get_reg(train_X, validate_X, train_y, validate_y):

    # from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(solver='liblinear')
    logit.fit(train_X, train_y)

    # get model accuracy for training data
    print(f"Accuracy of Logistic Regression on train is {logit.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(validate_X, validate_y)}")

def get_knn(train_X, validate_X, train_y, validate_y):

    knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')

    knn.fit(train_X, train_y)

    print(f"Accuracy of Logistic Regression on train is {knn.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {knn.score(validate_X, validate_y)}")

def get_tree_test(train_X, test_X, train_y, test_y):

    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5, random_state=123)

    #fit model on training data
    clf = clf.fit(train_X, train_y)

    # get model accuracy for Decision Tree on test
    print(f"Accuracy of Decision Tree on test data is {clf.score(test_X, test_y)}")