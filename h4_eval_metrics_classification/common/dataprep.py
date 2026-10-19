from sklearn.model_selection import train_test_split


def data_overview(data_frame):
    print(data_frame.info())
    print('.......................................')
    data_frame.columns = data_frame.columns.str.lower().str.replace(' ', '_')
    check_missing(data_frame)
    print('.......................................')

def check_missing(data_frame):
    missing_sum = data_frame.isnull().sum()
    print(f"Missing values in each column:\n{missing_sum}")

def fill_missing(data_frame):
    categorical_cols = data_frame.select_dtypes(include=['object', 'string']).columns
    data_frame[categorical_cols] = data_frame[categorical_cols].fillna('NA')
    numerical_cols = data_frame.select_dtypes(include=['float64', 'int64']).columns
    data_frame[numerical_cols] = data_frame[numerical_cols].fillna(0.0)
    print(f"Recheck columns if has any missing values after filling:")
    print('.......................................')
    check_missing(data_frame)
    print('.......................................')
    return data_frame

