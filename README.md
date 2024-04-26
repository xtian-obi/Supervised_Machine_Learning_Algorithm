# Supervised Machine Learning Algorithm
## INTRODUCTION
 Most of the issues  with the lending industry is not about the availability of  funds to give out its more about the  willingness of people to return or repay the loan given to them at the appointed time and with the agreed interest rate, this has resulted in a lot of issue in the lending industry making it hard for people who have money to give, not interested in lending.
The aim of this project is to develop a  machine learning model that will learn from previous data on the behaviour of previous borrowers who have been given loans and how they reacted after, this project aim to help lender predict if an borrower requesting for loan will pay back or not drawing conclusion from the borrower’s details and comparing it with previous behaviour of people who posses similar details and characteristics 

For this project we will be exploring publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
Lending club had a very interesting year in 2016, so let's check out some of their data and keep the context in mind. This data is from before they even went public.
We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from here or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
Here are what the columns represent:
* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* instalment: The monthly instalments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilisation rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

This project aim to help the lending industry reduce loss to borrowers who will not payback by almost accurately predicting their behaviours and decide wether or not a prospecting borrower should be awarded the loan making the lending industry free flowing and reducing loss.

## DATA PREPROCESSING
For the data preprocessing the data was relatively clean having no missing values, the categorical features were in the ‘purpose’ columns since there was not a lot of unique values in the column encoded the categorical features by putting them in a dictionary with the categorical features as the key and  corresponding integral as the values

## EXPLORATORY DATA ANALYSIS (EDA)
With the help of matplotlib and seaborn data visualisation and exploration was done:
With matplotlib histogram of the ‘fico’ column and also for the two possible credit policy was obtain given insight of the behaviour of borrowers with different ‘fico’ points, same was done for the ‘fico’ column  with those that paid and those that did not pay.
With Seaborn a count plot of the ‘purpose’ column was done between those that paid and those that did not pay for different reasons of borrowing.
With seaborn also a joint plot was obtained for the ‘fico’ column against the ‘int.rate’, and lastly with seaborn a lmplot was done for the’fico’ column against the ‘int.rate’ coloured by credit policy for those that paid back and those that did not pay back.

## MODELLIING APROACH 
A couple of models were considered for this task like the DecisionTreeClassifier and th RandomForestClassifier this models were selected because the model is to perform a classification task choosing between wether an event occurred or not, Radom forest classifier and Decision Tree Classifier are  two models that have shown great accuracy in this type of classification and it can be fine tuned to improve the accuracy
For the Evaluation metric the classification report and the confusion matrix was chosen to evaluate the model’s performance
Confusion Matrix: It is a table that is often used to evaluate the performance of a classification model. It compares the actual labels of a dataset to the labels predicted by the model.
The confusion matrix has four main components:
1. True Positive (TP): The model correctly predicts the positive class.
2. True Negative (TN): The model correctly predicts the negative class.
3. False Positive (FP): Also known as Type I error, the model incorrectly predicts the positive class when the actual class is negative.
4. False Negative (FN): Also known as Type II error, the model incorrectly predicts the negative class when the actual class is positive.
Classification Report:It is a summary of the performance of a classification model. It provides key evaluation metrics for each class in a classification problem.
Typically, a classification report includes the following metrics for each class:
1. Precision: The proportion of true positive predictions out of all positive predictions made by the model.
2. Recall: The proportion of true positive predictions out of all actual positive instances in the dataset.
3. F1-score: The harmonic mean of precision and recall, providing a balance between the two metrics.
4. Support: The number of actual occurrences of each class in the dataset.
Additionally, the classification report often includes an overall accuracy score for the entire model.
By examining the classification report, we can gain insights into how well the model performs for each class and identify any imbalances or areas for improvement in the model's predictions.

## MODEL DEVELOPMENT
The train test split was imported  to split the data into training set and testing set  before fitting the training split of the data into the algorithm 
For DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                                             presort=False, random_state=None, splitter='best')
The following hyper-parameter tuning  was used for the DecisionTreeClassifier
For RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                                 min_samples_leaf=1, min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
                                                 oob_score=False, random_state=None, verbose=0,
                                                 warm_start=False)
The following hyper-parameter tuning  was used for the RandomForestClassifier

## MODEL EVALUATION
For the DecisionTreeClassifier the classification report  brought out the following report  
        precision    recall  f1-score   support

          0       0.85      0.82      0.84      2431
          1       0.19      0.23      0.20       443

avg / total       0.75      0.73      0.74      2874
 And the confusion matrix brought 
[[1995  436]
 [ 343  100]]

For the RandomForestClassifier  the classification report
     precision    recall  f1-score   support

           0       0.84      0.99      0.91      2650
           1       0.36      0.02      0.03       511

    accuracy                                 0.84      3161
   macro avg       0.60      0.51    0.47      3161
weighted avg      0.76     0.84    0.77      3161

And for confusion matrix
[[2634   16]
 [ 502    9]]

For higher accuracy RandomForestClassifie was chosen for the model since it brought forward a greater accuracy

## CONCLUSION
The RandomForestClassifier model achieved an overall accuracy of 84%, demonstrating its effectiveness in classifying the majority class. However, its performance varied across classes, with higher precision and recall for class 0(those that have not fully paid) compared to class 1(those that have paid).



[![Launch App](https://img.shields.io/badge/Launch-App-brightgreen?style=for-the-badge)](https://share.streamlit.io/xtian-obi/Supervised_Machine_Learning_Algorithm/main/lab phase Supervised Machine learnin Algorith.py)
