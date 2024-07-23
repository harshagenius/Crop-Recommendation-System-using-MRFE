# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Custom transformer for Modified Recursive Feature Elimination
class ModifiedRFE(TransformerMixin):
    def __init__(self, estimator, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        self.rfe = RFE(self.estimator, n_features_to_select= 7)#self.n_features_to_select
        self.rfe.fit(X, y)
        return self

    def transform(self, X):
        return self.rfe.transform(X)

if __name__=="__main__":
    # Load dataset
    data = pd.read_csv("Crop_recommendation.csv")
    print(data.head())
    print(data.shape)
    data.info()
    print(data.isnull().sum()) 


    print(data.duplicated().sum())
    print(data.describe())

    # Assuming 'data' is your DataFrame
    numeric_data = data.select_dtypes(include='number')  # Select only numeric columns
    co_re = numeric_data.corr()  # Calculate correlation matrix


    #import seaborn as sns
    print(sns.heatmap(co_re,annot=True,cbar=True,cmap='coolwarm'))

    data['label'].value_counts()

    #import matplotlib.pyplot as plt
    sns.displot(data['N'])
    plt.show()
    # Define crop mapping
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
        'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
        'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }

    # Map crop names to numeric labels
    data['crop_num'] = data['label'].map(crop_dict)
    print(data.head())

    # Prepare features and target
    X = data.drop(['crop_num', 'label'], axis=1) 
    y = data['crop_num']

    print(f" X : {X}")
    print(f"y : {y.shape}")
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)

    # Preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('mrfe', ModifiedRFE(estimator=DecisionTreeClassifier(), n_features_to_select=7)),])
       # ('minmax', MinMaxScaler()),
       # ('std', StandardScaler())
    #])
    # minmaxscaler [added]
    mx=MinMaxScaler()
    X_train=mx.fit_transform(X_train)
    X_test=mx.transform(X_test)
    
    # standard scaler [added]
    sc=StandardScaler()
    sc.fit(X_train)
    X_train=sc.transform(X_train)
    X_test=sc.transform(X_test)
    # Fit and transform training data
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train,y_train)#preprocessing_pipline.fit_transform(X_train, y_train)

    # Transform test data
   # X_train_preprocessed=preprocessing_pipeline.transform(X_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    # Dictionary of classifiers
    models = {
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'Bagging': BaggingClassifier()
    }

    # Train and evaluate classifiers
    for name, model in models.items():
        model_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', model)
        ])
        model_pipeline.fit(X_train_preprocessed,y_train)
        y_pred = model_pipeline.predict(X_test_preprocessed)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
     
     
    # Soft Voting Classifier
    voting_classifier = VotingClassifier(estimators=[(name, Pipeline([('preprocessing', preprocessing_pipeline),
                                                                    ('classifier', model)])) for name, model in models.items()],
                                        voting='soft')
    voting_classifier.fit(X_train_preprocessed, y_train)
    y_pred_voting = voting_classifier.predict(X_test_preprocessed)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    precision_voting = precision_score(y_test, y_pred_voting, average='weighted')
    recall_voting = recall_score(y_test, y_pred_voting, average='weighted')
    f1_voting = f1_score(y_test, y_pred_voting, average='weighted')
    print(f"Soft Voting Classifier - Accuracy: {accuracy_voting}, Precision: {precision_voting}, Recall: {recall_voting}, F1 Score: {f1_voting}")

    # Save model and preprocessors
    pickle.dump(voting_classifier, open('model.pkl', 'wb'))
    pickle.dump(mx,open('minmaxscaler.pkl','wb'))
    pickle.dump(sc,open('standscaler.pkl','wb'))
    pickle.dump(preprocessing_pipeline, open('preprocessing_pipeline.pkl', 'wb'))

        
