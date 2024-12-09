import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Print column names to debug
    print("Columns in the dataset:", data.columns)
    
    # Preprocess data
    data = data.dropna()  # Drop rows with missing values
    
    # Define features (X) and target variable (y)
    X = data[['Temperature', 'CO2', 'Humidity']]  # Replace with actual column names from the dataset
    y = data['Agriculture_Variable']  # Replace with the actual target variable column name
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
