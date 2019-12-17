"""
Created on Thu Dec 12 09:50:53 2019

@author: ABDEDDAIM Omar
 Subject : Lineaire Regression of Steel fatigue.
"""

# Standard library imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

### Matplotlib and styling
%matplotlib inline
sns.set_context("poster", font_scale=1.1)
data = pd.read_csv("data.csv")

# Define scaler
#scaler = RobustScaler()
scaler = StandardScaler()

# Create a dataframe with all input values
X = pd.DataFrame(scaler.fit_transform(data.drop(['Fatigue'], axis=1)))
X.columns = data.drop(['Fatigue'], axis=1).columns
# Similarly, extract the y values as a vector
y = data['Fatigue']

# Initialize model tracking variables
rmses, r2s, results, predicted, actual = ([] for i in range(0, 5))
n_estimators = 200    # Set number of estimators
# Set number of folds 15
num_tests = 50
learning_rate = 0.075
# Set up validation using kFolds
k_fold = KFold(n_splits=15, shuffle=True)
k_fold.split(X)
print(k_fold)

for i in range(0, num_tests):
    if not i % 10:
        print("Test %s" % (i))
    # Train and test each fold, and track results
    for k, (train, test) in enumerate(k_fold.split(X)):
        # Initialize model
        model_name, model = ("Gradient Boosting Regression %s" % (n_estimators),
                             GradientBoostingRegressor(n_estimators=n_estimators,
                                                       learning_rate=learning_rate,
                                                       max_depth=4, loss='ls'))

        # Fit the model on the kfold training set and predict the kfold test set
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])

        # Save r^2 and root mean squared error for each fold
        r2s.append(r2_score(y.iloc[test], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test], pred)))

        # Save predictions vs actual values for later plotting
        predicted.append(pred)
        actual.append(y.iloc[test])
print("***** %s - %s tests *****" % (model_name, num_tests))
print("rMSE:%s" % (np.mean(rmses)))
print("r^2:%s" % (np.mean(r2s)))

fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter([item for sublist in actual for item in sublist], [item for sublist in predicted for item in sublist],
           s=75, c='blue', alpha=0.05)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1200)
ax.set_xlabel('Actual Fatigue Strength')
ax.set_ylabel('Predicted Fatigue Strength')
x = np.linspace(*ax.get_xlim())
ax.plot(x, x, linewidth=6,  alpha=0.1)
sns.despine()
