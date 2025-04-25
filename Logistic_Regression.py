# Load libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score)


 
def logistc_stability_pred(data_path, test_size=0.2):
    
    """
    Predict grid stability using logistic regression.

    This function trains a model to classify grid stability based on input features. 
    It processes the dataset, encodes the target column, scales the features, trains 
    a logistic regression model and evaluates it using accuracy and a classification report.

    Parameters:
        data_path (str): Path to the CSV file containing the dataset.
        test_size (float, optional): Proportion of data for testing (default is 0.2).

    Prints:
        - Model accuracy (2 decimal places).
        - Classification report (precision, recall, F1-score).
    """ 

    # load the data
    stability_df = pd.read_csv(data_path)

    # Initialize the LabelEncoder
    encoder = LabelEncoder()

    # Fit and transform the Target column
    stability_df['stabf'] = encoder.fit_transform(stability_df['stabf'])


    # split the data into training and testing
    X = stability_df.drop(['stabf', 'stab', 'p1', 'p2', 'p3', 'p4'], axis =1)
    y = stability_df['stabf']
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=test_size, random_state=42)
    
    # Instantiate standard scaler 
    scaler = StandardScaler()

    # scale X_train and X_test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled =scaler.transform(X_test)

    # Instantiate the model
    model = LogisticRegression()

    # fit the model 
    model.fit(X_train_scaled, y_train)

    # predict the y_values
    y_pred = model.predict(X_test_scaled) 

    # Evaluation metrics
    accuarcy = accuracy_score(y_pred, y_test)
    class_report = classification_report(y_pred, y_test)

    # Print evaluation metrics 
    print(f"The accuarcy score: {accuarcy:.2f}")
    print(f" Classification report....... \n {class_report}")
    
# call the function 
logistc_stability_pred('stability_dataset.csv')
