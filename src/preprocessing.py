# cleaning and preprocessing the data before it is used in the machine learning algorithms
def clean_data(df):
    # Remove null values
    df = df.dropna()
    
    # Remove outliers
    df = df[df['column_name'] < df['column_name'].quantile(0.75)]
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['categorical_column_1', 'categorical_column_2'])
    
    return df
