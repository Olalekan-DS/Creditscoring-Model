# to handle datasets
import pandas as pd
import numpy as np
pd.pandas.set_option('display.max_columns', None)
import seaborn as sns #Graph library that use matplot in background
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

#plotly libary
import plotly.offline as py 
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
from collections import Counter # To do counter of some features

import warnings
warnings.filterwarnings('ignore')

#Scaling
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#############################################################################################################
df = pd.read_csv('/Users/user/Documents/creditscore_new.csv')
df.shape
print(df.info())
df.isnull().sum()

#EDA
#Let's start looking through target variable and their distribuition
trace0 = go.Bar(
            x = df[df["Risk_new"]== 'good']["Risk_new"].value_counts().index.values,
            y = df[df["Risk_new"]== 'good']["Risk_new"].value_counts().values,
            name='Good credit'
    )
trace1 = go.Bar(
            x = df[df["Risk_new"]== 'medium']["Risk_new"].value_counts().index.values,
            y = df[df["Risk_new"]== 'medium']["Risk_new"].value_counts().values,
            name='Medium credit'
    )
trace2 = go.Bar(
            x = df[df["Risk_new"]== 'bad']["Risk_new"].value_counts().index.values,
            y = df[df["Risk_new"]== 'bad']["Risk_new"].value_counts().values,
            name='Bad credit'
    )

data = [trace0, trace1, trace2]

layout = go.Layout()

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Risk Variable'
    ),
    title='Target variable distribution'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')

#Looking at the distribuition of loan status by Risk level
#First plot
trace0 = go.Bar(
    x = df[df["Risk_new"]== 'good']["loan_status"].value_counts().index.values,
    y = df[df["Risk_new"]== 'good']["loan_status"].value_counts().values,
    name='Good credit'
)
#Second plot
trace1 = go.Bar(
    x = df[df["Risk_new"]== 'medium']["loan_status"].value_counts().index.values,
    y = df[df["Risk_new"]== 'medium']["loan_status"].value_counts().values,
    name='Medium credit'
)
#third plot
trace2 = go.Bar(
    x = df[df["Risk_new"]== 'bad']["loan_status"].value_counts().index.values,
    y = df[df["Risk_new"]== 'bad']["loan_status"].value_counts().values,
    name="Bad Credit"
)

data = [trace0, trace1, trace2]

layout = go.Layout(
    title='Loan Status Distribuition'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Loan_Status-Grouped')

#Data Distribution by partner
df_good = df.loc[df["Risk_new"] == 'good']['partner_name'].values.tolist()
df_medium = df.loc[df["Risk_new"] == 'medium']['partner_name'].values.tolist()
df_bad = df.loc[df["Risk_new"] == 'bad']['partner_name'].values.tolist()
df_partner = df['partner_name'].values.tolist()

#First plot
trace0 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)
#Second plot
trace1 = go.Histogram(
    x=df_medium,
    histnorm='probability',
    name="Medium Credit"
)
#Third plot
trace2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)
#Fourth plot
trace3 = go.Histogram(
    x=df_partner,
    histnorm='probability',
    name="Partner"
)

data = [trace0, trace1, trace2]

layout = go.Layout(
    title='General Distribuition'
)

fig = go.Figure(data=data, layout=layout)
fig['layout'].update(showlegend=True, title='Partner Distribuition', bargap=0.05)
py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')

#Cross tabulating anount to pay with partner
fig, ax = plt.subplots(figsize=(20,20), nrows=2)
g1 = sns.boxplot(x="partner_name", y="balance_remaining", data=df, palette="hls", ax=ax[0], hue="Risk_new")
g1.set_title("Balance Remaining by Partner", fontsize=12)
g1.set_xlabel("Partner", fontsize=12)
g1.set_ylabel("Credit Amount", fontsize=12)

g2 = sns.violinplot(x="call_contact_status", y="balance_remaining", data=df, ax=ax[1], hue="Risk_new", palette="hls")
g2.set_title("Credit Amount by Call", fontsize=12)
g2.set_xlabel("Call", fontsize=12)
g2.set_ylabel("Credit Amount", fontsize=12)

plt.subplots_adjust(hspace = 0.4,top = 0.9)
plt.show()

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
