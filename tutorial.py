import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('winequality.csv')
print(df.head())

df.info()

print(df.describe().T)

print("\n\nEDA - check for nulls")
print(df.isnull().sum())

print("\n\nimpute the missing values by means")
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())

print("\n\nnow the nulls should be zero")
print(df.isnull().sum().sum())

# histogram to visualise the distribution of the data with continuous values in the columns of the dataset.
df.hist(bins=20, figsize=(10, 10))
#plt.show()

# Now let's draw the count plot to visualise the number data for each quality of wine.
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
#plt.show()

no_type_df = df.copy()
no_type_df = no_type_df.drop('type', axis=1)

plt.figure(figsize=(12, 12))
sb.heatmap(no_type_df.corr() > 0.7, annot=True, cbar=False)
#plt.show()

# the last visualisation shows that there is a high correlation between 'free sulfur dioxide' and 'total sulfur dioxide'.
# So we can drop one of these columns to avoid redundancy because it does not help by increasing the model's performance
df = df.drop('total sulfur dioxide', axis=1)

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

df.replace({'white': 1, 'red': 0}, inplace=True)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

print(xtrain.shape, xtest.shape)

# Normalizing
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()

disp = ConfusionMatrixDisplay.from_estimator(
    models[1],
    xtest,
    ytest,
)
# Optional: Add a title and display the plot
disp.ax_.set_title("Confusion Matrix") #
#plt.show()

print(metrics.classification_report(ytest,
                                    models[1].predict(xtest)))

joblib.dump(models[1], 'wine_quality_model.pkl')
joblib.dump(norm, 'scaler.pkl')
print("Model and scaler saved")

loaded_model = joblib.load('wine_quality_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame({
    'type': [1, 0],  # 1 for white, 0 for red (two samples)
    'fixed acidity': [7.0, 8.5],
    'volatile acidity': [0.3, 0.5],
    'citric acid': [0.4, 0.2],
    'residual sugar': [2.0, 1.5],
    'chlorides': [0.05, 0.08],
    'free sulfur dioxide': [20.0, 15.0],
    'density': [0.995, 0.997],
    'pH': [3.2, 3.4],
    'sulphates': [0.6, 0.7],
    'alcohol': [10.5, 11.0]
})

# Preprocessing: Drop 'total sulfur dioxide' (already done in this sample), encode 'type' (already 0/1), scale
new_data_scaled = loaded_scaler.transform(new_data)
new_predictions = loaded_model.predict(new_data_scaled)

print("Predictions for new data (0=not best quality, 1=best quality):", new_predictions)
