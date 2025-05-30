import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(path):
    # loading the dataset
    df = pd.read_csv(path)

    # converting TotalCharges to numeric values and setting NaNs to 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # converting columns with object dtypes to numeric type
    boolean_column = ["Partner", "Dependents", "PaperlessBilling", "Churn", "PhoneService"]
    multi_category_column = ["MultipleLines", "InternetService", "OnlineSecurity", 
                            "OnlineBackup", "DeviceProtection", "TechSupport", 
                            "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
    
    df[boolean_column] = df[boolean_column].replace({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].replace({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=multi_category_column, drop_first=True)

    # dropping the customer ID column as they are just identifiers and not features
    df = df.drop('customerID', axis=1)

    # setting up features and labels
    X = df.drop('Churn', axis=1)
    y = df["Churn"]

    #splitting the dataset for train, val and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train) 

    # normalizing the dataset 
    numerical_column = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_column] = scaler.fit_transform(X_train[numerical_column])
    X_val[numerical_column] = scaler.transform(X_val[numerical_column])
    X_test[numerical_column] = scaler.transform(X_test[numerical_column])

    return X_train, X_val, X_test, y_train, y_val, y_test




