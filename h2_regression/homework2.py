from contextlib import redirect_stdout
import pandas as pd

from h2_regression.common.data_overview import car_data_overview, select_columns, median_horsepower_calc, \
    data_prep_split_42, plot_fuel_efficiency, data_split_seeds_selection, data_split_seed_nine

import sys
from contextlib import redirect_stdout
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./data/car_fuel_efficiency_2026.csv")
    df = data.copy()

    # 3. Open output file safely
    with open("results/homework2_output.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Data overview")
            print("..............................")
            car_data_overview(df)
            plot_fuel_efficiency(df)
            dff = df.copy()
            sel_df = select_columns(dff)
            median_horsepower_calc(sel_df)
            # seed 42
            df_42 = sel_df.copy()
            data_prep_split_42(df_42)
            # seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            df_seedz =sel_df.copy()
            data_split_seeds_selection(df_seedz)
            # seed 9
            df_nine = sel_df.copy()
            data_split_seed_nine(df_nine)
