import streamlit as st
#imported all the necessary libraries needed to load the data set explore it and do some visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#used the pandas library to load the dataset

data=pd.read_csv('loan_data.csv')

df = data.copy()

df.head(5)

df.info()
df.describe()

#used matplotlib to display histogram of the fico column for the two possible credit policy

plt.figure(figsize=(12,7))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='green',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

#used matplotlib to display histogram of the fico column for the two possible outcome of those that paid and those that didn't

plt.figure(figsize=(12,7))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='green',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#used seaborn to display a countplot of the purpose column for the those that paid and those that didn't paid

plt.figure(figsize=(12,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')

#used seaborn to display a jointplot of the fico column and the int.rate column

sns.jointplot(x='fico',y='int.rate',data=df,color='orange')

#used seaborn to display a lmplot of fico and int.rate column between thos that paid and those that didn't pay
#coloring the plot based on the credit policy

plt.figure(figsize=(12,7))
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')

df.info()

key = data.purpose.unique()
value = list(range(len(key)))
purop = dict(zip(key,value))

df['purpose'] = df.purpose.map(purop)

#imported train_test_split from sklearn in order to split the data into test set and training set

from sklearn.model_selection import train_test_split

X=df.drop('not.fully.paid', axis=1)
y=df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#imported RandomForestClassifier from sklean another algorithm to further train the model to compare its performance

from sklearn.ensemble import RandomForestClassifier

rc=RandomForestClassifier(n_estimators=600)

rc.fit(X_train,y_train)

pred=rc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))



data.head(1)


with st.form('Enter basic details Below'):
    # Streamlit form
    st.title('Loan Prediction')

    # Input fields
    credit_policy = st.number_input('credit policy', min_value=data['credit.policy'].min(),
                                    max_value=data['credit.policy'].max())
    purpose = st.selectbox('Select a Purpose', data['purpose'].unique())
    interest_rate = st.number_input('Interest rate', min_value=data['int.rate'].min(), max_value=data['int.rate'].max())
    installment = st.number_input('Installment', min_value=data['installment'].min(),
                                  max_value=data['installment'].max())
    annual_income = st.number_input('Annual Income', min_value=data['log.annual.inc'].min(),
                                    max_value=data['log.annual.inc'].max())
    dti = st.number_input('DTI', min_value=data['dti'].min(), max_value=data['dti'].max())
    fico = st.number_input('FICO', min_value=data['fico'].min(), max_value=data['fico'].max())
    days = st.number_input('Days', min_value=data['days.with.cr.line'].min(), max_value=data['days.with.cr.line'].max())
    revol = st.number_input('Revol', min_value=data['revol.bal'].min(), max_value=data['revol.bal'].max())
    revol_util = st.number_input('Revol Util', min_value=data['revol.util'].min(), max_value=data['revol.util'].max())
    last_6months = st.number_input('Last 6 months', min_value=data['inq.last.6mths'].min(),
                                   max_value=data['inq.last.6mths'].max())
    delinq_2yrs = st.number_input('Delinq 2 years', min_value=data['delinq.2yrs'].min(),
                                  max_value=data['delinq.2yrs'].max())
    pub_rec = st.number_input('Pub rec', min_value=data['pub.rec'].min(), max_value=data['pub.rec'].max())

    # Check if the form is submitted
    submit = st.form_submit_button('Click Here for Loan Preiction')

    if submit:
        purpose = purop[purpose]
        data = np.array([credit_policy, purpose, interest_rate, installment, annual_income, dti, fico,days,revol,revol_util,last_6months,delinq_2yrs,pub_rec])

        # Reshape the input array to have one row and multiple columns
        data_reshaped = data.reshape(1, -1)
        st.write('RESHAPED : - ', data_reshaped)
        # Now, you can use the reshaped data for prediction
        predict = rc.predict(data_reshaped)
        predict =predict [0]
        main = 'Fully Paid' if predict == 0 else 'Not Fully Paid'

        st.write('Loan Outcome is',main)
