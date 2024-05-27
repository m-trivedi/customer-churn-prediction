from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ChurnPredictor:
    def __init__(self):
        self.xcols = ['MonthlyCharges', 'TotalCharges', 'PaperlessBilling', 'PhoneService', 'tenure', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 'InternetService', 'MultipleLines', 'Partner', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'Dependents']
        self.model = Pipeline([
            ('rf', RandomForestClassifier(n_estimators=14, random_state=42)),
        ])
        
    def fit(self, train_df):
        train_df = self.add_stats(train_df)
        self.model.fit(train_df[self.xcols], train_df["Churn"])
        scores = cross_val_score(self.model, train_df[self.xcols], train_df["Churn"])
        print(f"AVG: {scores.mean()}, STD: {scores.std()}")

        rf_classifier = self.model.named_steps['rf']

        # Get feature importances
        importances = rf_classifier.feature_importances_
        
        # Print the feature ranking
        print("Feature ranking:")
        for i, importance in enumerate(importances):
            print(f"Feature {i}: {importance}")
    
    def predict(self, test_df):
        test_df = self.add_stats(test_df)
        return self.model.predict(test_df[self.xcols])

    def add_stats(self, data): # helper function
        
        mydf = pd.get_dummies(data[['Contract']], dtype=int)

        data['Contract_Month-to-month'] = mydf['Contract_Month-to-month']
        data['Contract_One year'] = mydf['Contract_One year']
        data['Contract_Two year'] = mydf['Contract_Two year']

        mydf2 = pd.get_dummies(data[['PaymentMethod']], dtype=int)

        for col in ['PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']:
            data[col] = mydf2[col]
        
        return data
    