import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import pickle

def create_model(data):
    X = data.drop('label', axis=1)
    y = data['label']

    #Scale the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    #Train-Test Split (80% train, 20% test) with stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Create XGBoost model for multiclass classification
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=22, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for precision in multiclass
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for recall in multiclass
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for F1 score in multiclass
    classification = classification_report(y_test, y_pred)

    
    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Classification report: {classification}")

    return model, scaler


def get_data_clean():
    data = pd.read_csv("data/data.csv")
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data

def main():
    data = get_data_clean()
    model, scaler = create_model(data)

    with open("model/pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/pkl", "wb") as f:
        pickle.dump(scaler, f)
    


if __name__== '__main__':
    main()
