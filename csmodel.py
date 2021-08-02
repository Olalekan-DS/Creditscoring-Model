# to handle datasets
import pandas as pd
import numpy as np
pd.pandas.set_option('display.max_columns', None)
# for plotting
import matplotlib.pyplot as plt
#%matplotlib inline
# to divide train and test set
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
from math import sqrt

import warnings
warnings.filterwarnings('ignore')

#Scaling
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/user/Documents/creditscore_new.csv')
df.shape

# Select independent and dependent variable
include = ['partner_name', 'days_late',
       'call_contact_status', 'type', 'status', 
       'balance_remaining', 'amount_already_paid', 'CreditStatus'] # Only four features
df_ = df[include]

#Transforming the data into Dummy variables
df_['type'] = df_['type'].fillna('no_inf')
df_['status'] = df_['status'].fillna('no_inf')
df_['days_late'].fillna(df_["days_late"].mean(), inplace = True)
df_['amount_already_paid'].fillna(df_["amount_already_paid"].mean(), inplace = True)

X = df_.loc[:,['partner_name','days_late','call_contact_status','type','status','balance_remaining', 'amount_already_paid']]
y = df_.CreditStatus


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X.loc[:,['partner_name','days_late','call_contact_status','type','status','balance_remaining', 'amount_already_paid']]= \
X.loc[:,['partner_name','days_late','call_contact_status','type','status','balance_remaining', 'amount_already_paid']].apply(enc.fit_transform)

X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(n_estimators=80,max_depth=11)
model1.fit(X_train,y_train)
y_predict1 = model1.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score
print("MAE : ", mean_absolute_error(y_test,y_predict1))
print(r2_score(y_test,y_predict1))

# Save your model
import joblib
joblib.dump(model1, 'creditscore_predict_model.pkl')
#print("Model dumped!")

# Load the model that you just saved
model1 = joblib.load('creditscore_predict_model.pkl')
print(model1)
print(model1.predict([[10, 415, 3, 1, 6, 7298, 4000]]))
