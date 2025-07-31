from sklearn.model_selection import train_test_split

def extract_features(df):
    '''Extract features and split into train/test.'''
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)
