import numpy as np
import pandas as pd

from h1_intro_ml.common.describe_data import print_overview, max_fuel_efficiency_asia, \
    overview_horsepower, sum_of_weights

if __name__ == '__main__':
    data = pd.read_csv('data/car_fuel_efficiency_2026.csv')
    df = data
    from contextlib import redirect_stdout
    import pandas as pd

    if __name__ == '__main__':
        data = pd.read_csv('data/car_fuel_efficiency_2026.csv')
        df = data
        with open("homework1_output.txt", "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print('Data overview')
                print('..............................')
                print_overview(df)
                overview_horsepower(df)
                max_fuel_efficiency_asia(df)
                sum_of_weights(df)
