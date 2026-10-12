from contextlib import redirect_stdout
import pandas as pd

from h3_classification.common.classification_methods import implement_classification_analysis
from h3_classification.common.data_analysis import course_conversion_data_overview, fill_missing_values, check_mode, \
    correlation_matrix_course_conversion, split_data_no_converted_anaylse

if __name__ == "__main__":
    data_course = pd.read_csv("./data/course_lead_scoring.csv")
    df = data_course

    with open("./results/homework3_output.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Data overview")
            print("..............................")
            course_conversion_data_overview(df)
            filled_df = fill_missing_values(df)
            check_mode(filled_df)
            correlation_matrix_course_conversion(filled_df)
            implement_classification_analysis(filled_df)
            # split_data_no_converted_anaylse(filled_df)

