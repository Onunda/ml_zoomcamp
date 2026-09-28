import pandas as pd
import numpy as np


def print_overview(data_frame):
    """ Print pandas data frame overview"""
    print('## Data frame info:')
    print(data_frame.info())
    print('\n')

    print('## Data head and tail:')
    print(data_frame.head(5))
    print('...')
    print(data_frame.tail(5))
    print('\n')

    print('## Data frame columns:')
    for column in data_frame.columns:
        print(column)
    print('******************* Homework one start ****************')
    print('Panda version : ', pd.__version__)
    print('...')
    print('\n')

    print('## Data frame shape:')
    print(f"Number of records in the dataset: {data_frame.shape[0]}")
    print('...')
    print(str(data_frame.shape[0]) + ' rows')
    print(str(data_frame.shape[1]) + ' columns')
    print('...')
    print('\n')

    # Number of unique fuel types
    unique_fuel_types = data_frame['fuel_type'].nunique()
    print(f"Number of unique fuel types: {unique_fuel_types}")
    print('...')
    print('\n')

    # Check for missing values and list the columns
    missing_cols_count = (data_frame.isnull().sum() > 0).sum()
    print(f"Number of columns with missing values: {missing_cols_count}")
    missing_values = data_frame.isnull().sum()
    print(f"Missing values in each column :\n{missing_values}")
    print('...')
    print(f"Columns with missing values and their counts:\n{missing_values[missing_values > 0]}")
    print('\n')


def max_fuel_efficiency_asia(data_frame):
    max_fuel_eff_asia = data_frame[data_frame['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
    print(f"Maximum fuel efficiency of cars from Asia: {max_fuel_eff_asia} MPG")
    print('...')


def overview_horsepower(data_frame):
    print(f"Median horsepower: {data_frame['horsepower'].median()}")
    print(f"Most frequent horsepower: {data_frame['horsepower'].mode()[0]}")
    print('...')
    frequent_horsepower = data_frame['horsepower'].mode()[0]
    data_frame['horsepower'] = data_frame['horsepower'].fillna(frequent_horsepower)
    print(f"Missing values in 'horsepower' after filling: {data_frame['horsepower'].isnull().sum()}")
    print('...')
    median_horsepower_fill_missing = data_frame['horsepower'].median()
    print(f"Median horsepower after filling missing values: {median_horsepower_fill_missing}")
    print('...')
    print('\n')


def sum_of_weights(data_frame):
    print('SUM OF WEIGHTS')
    cars_from_asia = data_frame[data_frame['origin'] == 'Asia']
    print('Display Asian cars')
    print('.........................................')
    print(cars_from_asia.head())
    print('Display Asian cars weight and model year')
    print('.........................................')
    asian_cars_weight_model_yr = cars_from_asia[['vehicle_weight', 'model_year']]
    print(asian_cars_weight_model_yr.head(7))
    print('\n')
    X = asian_cars_weight_model_yr.head(7).to_numpy()
    print(f"Shape of the NumPy array X:{X.shape}")
    print('.........................................')
    print(f'underlying NumPy array:{X}')
    print('.........................................')
    print('Transpose of X.T or matrix-matrix multiplication between the transpose of X and X:')
    XTX = X.T @ X
    print(f'XTX: \n {XTX}')
    XTX_inverse = np.linalg.inv(XTX)
    print(f'Inversion of XTX: \n {XTX_inverse}')
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    print(f'Array y:\n{y}')
    # Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w
    w = (XTX_inverse @ X.T) @ y
    print(f'result w: {w}')
    sum_w = np.sum(w)
    print(f'sum of all the elements of the result w : {sum_w}')
