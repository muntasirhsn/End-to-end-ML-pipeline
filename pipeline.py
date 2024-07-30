# Mute warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Import preprocessor, models
import preprocessor, models



# Import train data
train = pd.read_csv(f"data/train.csv", index_col="Id")
X_train = train.copy()
print('Train data size: ', X_train.shape)
y_train = X_train.pop("SalePrice")

# # Import test data
# test = pd.read_csv(f"data/test.csv", index_col="Id")
# X_test = test.copy()
# print('Test data size: ', X_test.shape)



# Root Mean Squared Logaritmic Error (RMSLE)
def rmsle(X, y, model):
    X = X.copy()
    score = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    score = score.mean()
    score = np.sqrt(score)
    return score


# End-to-end ML pipeline with Scikit-learn
# Put all the data processing steps into a Custom Transformer
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None

    def fit(self, X, y=None):
        X_transformed = self.transform(X)
        self.numeric_cols = X_transformed.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = X_transformed.select_dtypes(exclude=['number']).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X = preprocessor.preprocess(X)
        return X
    
# Custom data/feature preprocessor
data_preprocessor = DataPreprocessor()



# scaler for numeric features
scaler = RobustScaler()
# one-hot encoder for categorical features
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')



# Wrap ColumnTransformer to access dynamic feature names from the custom transformer
class ColumnTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.column_transformer = None

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        numeric_cols = self.preprocessor.numeric_cols
        categorical_cols = self.preprocessor.categorical_cols
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_cols),
                ('cat', ohe, categorical_cols)
            ],
            remainder = 'drop'
        )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

# Initialize the ColumnTransformerWrapper
column_transformer = ColumnTransformerWrapper(data_preprocessor)



# Full pipeline
pipeline = Pipeline(steps=[
    ('data_preprocessor', data_preprocessor),
    ('transform', column_transformer),
    ('stack_model', models.stack)
])



# Optional: Evaluate pipeline model. 
# Due to to the nature of the traget feature of the training data, we fit the model to its logarithmic value
# The example training data is very small and a cross-validation is necessary. 
# After the cross-validation, we need to fit the model again to train data before we can make predictions
print('\nModel evaluation in progress....')
score = rmsle(X_train, np.log1p(y_train), pipeline)
print("Root mean squared logarithmic (rmsle) error: {:.5f}".format(score), '\n')



# Fit pipeline to full training data
print('Fitting pipeline on training data')
pipeline.fit(X_train, np.log1p(y_train))
print('Complete!\n')



# # Optional: Make prediction on full test data. As the y was log-transformed during fitting/training the model, we need to exp-transform the predictions
# print('Running predictions')
# predictions = np.exp(pipeline.predict(X_test))
# output = pd.DataFrame({'Id': test.index, 'SalePrice': predictions})
# output.to_csv('y_pred.csv', index=False)
# print(output.head())