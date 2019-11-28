#!/usr/bin/env python
# coding: utf-8

## Data Mining Assignment: Classification Modelling
# Nina Kumagai (19389905)

## Part I (Data Preprocessing)
### 1.0 Exploratory Data Analysis (EDA):

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import normalize, PowerTransformer
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import arff


print("Pre-processing data...")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("data2019.student.csv")
#df.head()
#df.tail()
#print(df.describe())
#df.info()
#df.columns
#df.shape
#df.plot()

# As can be seen, values are not in the same ballpark.
#df.hist()
#df.groupby('Class').hist()
#df.index

# # 2.0 Data Preparation
# ## 2.1 Irrelevant Attributes
# ### What are Irrelevant Attributes?
# Irrelevant attributes are those which do not affect or influence the classification model. In other words these attributes do not help produce an optimal model for classification purposes - it may even inhibit such an optimal model from being built. They must be removed prior to analysis because Instance-based learning algorithms are intolerant of irrelevant attributes.

# ### Which Attributes are Irrelevant in Our Dataset?
# att14 and att17 is irrelevant because whether it is included in the analysis or not will not make a difference. This is because att14 and att17 both have the same response for every instance (as shown below) and thus will not make a difference to the classification model.

# for col in list(df):
#     print(col)
#     print(df[col].unique())

#df.att17.unique()
#df.att14.unique()

# Let's remove these irrelevant attributes now:
del df['att14']
del df['att17']

#df.columns
# Can confirm that att14 and att17 have been deleted as they do not appear in this list.

### 2.2 Missing Values:
df = df.replace(r'^\s*$', np.nan, regex=True)

# Missing Entries
df.isnull().sum()

# ### Which attributes have missing values?
# Attributes which have missing values include att3, att9, att25, att28, att13 and att19. 
#### For those attributes/instances, how many missing entries are present?
# It appears that there are 100 missing values in the attribute Class as expected. There are 4, 5, 3 and 6 missing values in att3, att9, att25 and att28 respectively. There are a large number of missing values in att13 and att19 with 1028 and 1034 missing values respectively.  

#### For each attribute/instance with missing entries, make a suitable decision, justify it, and proceed.
 
##### Small Number of Missing Data:
###### att3: 
# This is a categorical data column and there are only four missing values. Thus we will impute the values with the most frequent occuring category for this attribute.

###### att9: 
# Same as att3 since att9 is also a categorical data column with only 5 missing values.

###### att25: 
# This is a numerical data column and there are only 3 missing values. In order to fix this, the average values of the available data in att25 will be calculated and this average will be recorded in the three missing points.

###### att28: 
# Same as att25 since att28 is also numerical in nature and has only 6 missing values.
 
##### Large Number of Missing Data:
###### att13: 
# Categorical in nature. Remove the column as there are far too many missing values in order for us to impute or accept the missing values.

###### att19: 
# Numerical in nature. Remove the column as there are far too many missing values in order for us to impute or accept the missing values.

# Impute missing values with the mode of that column (mode is most frequently occuring value)
df['att3'].fillna(df['att3'].mode()[0], inplace=True)

# Check if imputation worked by counting number of missing values again:
# train_df.isnull().sum()

# att9
# Impute missing values with the mode of that column (mode is most frequently occuring value)
df['att9'].fillna(df['att9'].mode()[0], inplace=True)
# Check if imputation worked by counting number of missing values again:
# train_df.isnull().sum()

# att25
# Impute missing values with the mean of column
df['att25'].fillna(df['att25'].mean(), inplace=True)
# Check if imputation worked by counting number of missing values again:
# train_df.isnull().sum()

# att28
# Impute missing values with the mean of column
df['att28'].fillna(df['att28'].mean(), inplace=True)
# Check if imputation worked by counting number of missing values again:
# train_df.isnull().sum()

# Now delete att13 and att19
del df['att13']
del df['att19']
df.isnull().sum()

# ## 2.3 Duplicates
#### Detect if there are any duplicates (instances/attributes) in the original data?
# From simple observation we can confirm there are attribute duplicates. More in-depth check done below...
#### For each attribute/instance with duplicates, make a suitable decision, justify it, and proceed.

# Check for row duplicates:
len(df)

# dropping duplicate values 
df.drop_duplicates(keep=False,inplace=True) 
  
# length after removing duplicates 
len(df) 

# Thus there were no duplicate rows!

# Duplicate columns?
def getDuplicateColumns(df):
    '''Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.'''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)

# Get list of duplicate columns
duplicateColumnNames = getDuplicateColumns(df)
#print('Duplicate Columns are as follows')
#for col in duplicateColumnNames:
#    print('Column name : ', col)

# Delete duplicate columns
df = df.drop(columns=getDuplicateColumns(df))
# print("Modified Dataframe", train_df, sep='\n')

### Eliminate Redundant Columns
# Remove ID as it does not give us any information.
del df['ID']
df.head()

### Multicollinearity
df_numeric = df[['att18', 'att20', 'att21', 'att22', 'att25', 'att28']]
df_numeric.head()
df_numeric = df._get_numeric_data()
df_numeric.head()

from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#get_ipython().run_cell_magic('capture', '', '#gather features\nfeatures = "+".join(df_numeric.columns)\n\n# get y and X dataframes based on this regression:\ny, X = dmatrices(\'Class ~\' + features, df_numeric, return_type=\'dataframe\')')

# For each X, calculate VIF and save in dataframe
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#vif["features"] = X.columns
#vif.round(1)
# All the VIF factors are below 5 so there exists no multicollinearlity.

### 2.4 Data type:
#### For each attribute, carefully examine the default data type (e.g. Numeric, Nominal, Binary, String, etc.) that has been decided when Weka loads the original CSV file.

#df.info()
# Or we can also call dtypes.
#df.dtypes

# for col in list(df):
#     print(col)
#     print(df[col].unique())

#### If the data type of an attribute is not suitable, give a brief explanation and convert the attribute to a more suitable data type. Provide detailed information of the conversion.
##### Let's try to reclassify according to the output above:

# Note that object is a string in pandas, float64 is a decimal value, int64 is an integer.
# Note that categorical = nominal

# Attribute    Current     Reclassified 
# ID           int64       categorical
# Class       float64      categorical   (bivariate - binary) 0 or 1    BOOLEAN?
# att1         object      categorical
# att2         object      categorical
# att3         object      categorical
# att4         object      categorical
# att5         object      categorical
# att6         object      categorical
# att7         object      categorical
# att9         object      categorical
# att10        object      categorical
# att11        object      categorical
# att12        object      categorical
# att15         int64      categorical (bivariate - binary)   1 or 0    BOOLEAN?
# att16         int64      categorical (bivariate - binary)   1 or 2    BOOLEAN?
# att18         int64      numerical
# att20         int64      numerical
# att21         int64      numerical
# att22         int64      numerical
# att23         int64      categorical (bivariate - binary)   1 or 2     BOOLEAN?
# att25       float64      numerical - int64
# att26         int64      categorical    1,4,3,2 
# att27         int64      categorical    4,3,2,1
# att28       float64      numerical - int64
# att29         int64      categorical    1,2,3,4
# att30         int64      categorical    1,2,3

categ_list = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 
              'att9', 'att10', 'att11', 'att12']
#omitted att15 as it is 1,0 binary
#Same with att16 and att23
#Same with att26, att27, att29, att30

#### Let's use pandas dummy to produce new columns for the same column and make sure all categorical columns are numerical in nature (but not in a continuous hierarchial nature).
df.head()

for col in categ_list:
    df[col] = df[col].astype('category')

# for col in numerical_list:
#     train_df[col] = train_df[col].astype('int64')
# Now change class back to categ once it is int and not float.
df.dtypes

# train_df = pd.get_dummies(train_df, columns=[categ_list])
df = pd.get_dummies(df)
df.head()

df['att15'] = df['att15'].astype('category')
df['att16'] = df['att16'].astype('category')
df['att23'] = df['att23'].astype('category')

# att26, att27, att29, att30
df['att26'] = df['att26'].astype('category')
df['att27'] = df['att27'].astype('category')
df['att29'] = df['att29'].astype('category')
df['att30'] = df['att30'].astype('category')
df.dtypes

### Log Transformation
#df.hist(alpha=0.5, figsize=(20, 10))
#plt.tight_layout()
#plt.show()

numerical_list = ['att18', 'att20', 'att21', 'att22', 'att25', 'att28']
#df.hist(alpha=0.5, figsize=(20, 10), column=numerical_list)
#plt.tight_layout()
#plt.show()

# Looks like there are no outliers.

# plot original distribution plot
#fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
#ax1.set_title('Original Distributions')
#sns.kdeplot(df['att18'], ax=ax1)
#sns.kdeplot(df['att20'], ax=ax1)
#sns.kdeplot(df['att21'], ax=ax1)
#sns.kdeplot(df['att22'], ax=ax1)
#sns.kdeplot(df['att25'], ax=ax1)
#sns.kdeplot(df['att28'], ax=ax1);

# Let us perform log transformation on the data first.
# att18, 25 and 28 require a log transformation as it is skewed to the right!
df["att18"] = df["att18"].apply(np.log)
df["att25"] = df["att25"].apply(np.log)
df["att28"] = df["att28"].apply(np.log)

#df.hist(alpha=0.5, figsize=(20, 10), column=numerical_list)
#plt.tight_layout()
#plt.show()
# As can be seen, the values are standardised now with a mean of 0 and standard deviation of 1.

### Splitting the Dataset
# Training, Validation and Test. The last 100 rows are test. 
# The remaining dataset will be split 80% into training and 20% into validation (Cross validation will be implemented later when we repeat the sampling so we will only split training and test set here).

train_valid_df = df.iloc[:1000,]
test_df = df.iloc[1000:,]

#### For each numeric attribute, decide if any pre-processing (e.g. scaling, standardisation) is required. Give a brief explanation why it is needed (this should be discussed in relation to the subsequent classification task).
#print(len(train_valid_df))
#print(len(test_df))

### 2.5 Scaling and standardisation:
#### For each numeric attribute, decide if any pre-processing (e.g. scaling, standardisation) is required. Give a brief explanation why it is needed (this should be discussed in relation to the subsequent classification task).

numerical_list = ['att18', 'att20', 'att21', 'att22', 'att25', 'att28']
np.mean(train_valid_df.loc[:,numerical_list])
np.mean(test_df.loc[:,numerical_list])

# Means are similar, so use mean of training and apply to validation when standardising as more datapoints.
# Let us first standardise the numerical columns
# scaler = preprocessing.MinMaxScaler()  # we did not use min max scaler as we wanted centre to be 0.

# Mean is different, mean from the test for the standardisation of the test.
# Mean is consistent, use the mean from the training or the mean from the whole group of the dataset. 
scaler = preprocessing.StandardScaler()
train_valid_df.loc[:,numerical_list] = scaler.fit_transform(train_valid_df.loc[:,numerical_list].to_numpy())
# Use transform, not fit_transform to ensure it is same distribution as original
test_df.loc[:,numerical_list] = scaler.transform(test_df.loc[:,numerical_list].to_numpy())
train_valid_df.head()

# plot original distribution plot
#fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
#ax1.set_title('Original Distributions')
#sns.kdeplot(train_valid_df['att18'], ax=ax1)
#sns.kdeplot(train_valid_df['att20'], ax=ax1)
#sns.kdeplot(train_valid_df['att21'], ax=ax1)
#sns.kdeplot(train_valid_df['att22'], ax=ax1)
#sns.kdeplot(train_valid_df['att25'], ax=ax1)
#sns.kdeplot(train_valid_df['att28'], ax=ax1);

# plot original distribution plot
# fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
# ax1.set_title('Original Distributions')
# sns.kdeplot(train_valid_df['att20'], ax=ax1)
# sns.kdeplot(train_valid_df['att21'], ax=ax1)
# sns.kdeplot(train_valid_df['att22'], ax=ax1)
# sns.kdeplot(train_valid_df['att25'], ax=ax1)
# sns.kdeplot(train_valid_df['att28'], ax=ax1);

## Normalisation
# Let us now normalise the numerical columns
#normalise = preprocessing.normalize()
# train_df[numerical_list] = preprocessing.normalize(train_df[numerical_list].to_numpy())
# valid_df[numerical_list] = preprocessing.normalize(valid_df[numerical_list].to_numpy())
# test_df[numerical_list] = preprocessing.normalize(test_df[numerical_list].to_numpy())
# # plot original distribution plot
# fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
# ax1.set_title('Original Distributions')
# sns.kdeplot(train_df['att18'], ax=ax1)
# sns.kdeplot(train_df['att20'], ax=ax1)
# sns.kdeplot(train_df['att21'], ax=ax1)
# sns.kdeplot(train_df['att22'], ax=ax1)
# sns.kdeplot(train_df['att25'], ax=ax1)
# sns.kdeplot(train_df['att28'], ax=ax1);

## plot original distribution plot
# fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
# ax1.set_title('Original Distributions')
# sns.kdeplot(valid_df['att18'], ax=ax1)
# sns.kdeplot(valid_df['att20'], ax=ax1)
# sns.kdeplot(valid_df['att21'], ax=ax1)
# sns.kdeplot(valid_df['att22'], ax=ax1)
# sns.kdeplot(valid_df['att25'], ax=ax1)
# sns.kdeplot(valid_df['att28'], ax=ax1);

## plot original distribution plot
# fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
# ax1.set_title('Original Distributions')
# sns.kdeplot(test_df['att18'], ax=ax1)
# sns.kdeplot(test_df['att20'], ax=ax1)
# sns.kdeplot(test_df['att21'], ax=ax1)
# sns.kdeplot(test_df['att22'], ax=ax1)
# sns.kdeplot(test_df['att25'], ax=ax1)
# sns.kdeplot(test_df['att28'], ax=ax1);

# train_df.hist(alpha=0.5, figsize=(20, 10), column=numerical_list)
# plt.tight_layout()
# plt.show()
# valid_df.hist(alpha=0.5, figsize=(20, 10), column=numerical_list)
# plt.tight_layout()
# plt.show()
# test_df.hist(alpha=0.5, figsize=(20, 10), column=numerical_list)
# plt.tight_layout()
# plt.show()

# Note that you may need to apply a log transformation to att25, att28 since they are still skewed to the right! 

### 2.10 Others:
#### Describe other data-preparation steps not mentioned above.
# train_df_bal.sort_values('Class')
# test_df.sort_values('Class_0.0')
# df.groupby(level=0, sort=False).transform(lambda x: sorted(x,key=pd.isnull))
# This is to ensure null values of class is at bottom but not required anymore since we split to train set.

### 2.11 Save dataframe to csv to open in part II (Data Classification) Jupyter Notebook
### Save to arff format

arff.dump('training_set.arff'
      , train_valid_df.values
      , relation='relation name'
      , names=df.columns)

arff.dump('test_set.arff'
      , test_df.values
      , relation='relation name'
      , names=df.columns)

### Save to csv format
train_valid_df.to_csv('training_valid_set.csv', encoding='utf-8', index=False)
test_df.to_csv('test_set.csv', encoding='utf-8', index=False)


# # Part II (Data Classification)
# ## 2.11 Training, Validation, and Test Sets: 
# ### Suitably divide the prepared data into training, validation and test sets. These sets must be in ARFF format and submitted together with the electronic version of your report. See the Submission section for further information.

# ## 3.0 Classifier selection:
# You will need to select at least three (3) classifiers that have been discussed in the workshops: k-NN, Naive Bayes, and Decision Trees (J48). Other classifiers, including meta classifiers, are also encouraged. Every classifier typically has parameters to tune. If you change the default parameters to achieve higher cross-validation performance, clearly indicate what the parameters mean, and what values you have selected.

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree, metrics, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_valid_df = pd.read_csv('training_valid_set.csv')
train_valid_df.head()

### 3.1 Data imbalance:
#### The data set is known to have more samples from one class than the other. If you employ any strategy to address the data imbalance issue, describe it thoroughly.
# From general consensus, we have decided to balance the training set but leave the validation and test set untouched. This is so that we get the actual performance.
#train_valid_df.shape
train_valid_df['Class'].value_counts()

classes = train_valid_df['Class'].values
unique, counts = np.unique(classes, return_counts=True)
#plt.bar(unique,counts)
#plt.title('Class Frequency')
#plt.xlabel('Class')
#plt.ylabel('Frequency')
#plt.show()

# The simplest implementation of over-sampling is to duplicate random records from the minority class, which can cause overfitting. In under-sampling, the simplest technique involves removing random records from the majority class, which can cause loss of information.
# For our situation, we will implement over-sampling so we avoid data loss.

# Class count
#count_class_1, count_class_0 = train_valid_df['Class'].value_counts()
#print(count_class_1)

# Divide by class
#df_class_1 = train_valid_df[train_valid_df['Class'] == 1]
#df_class_0 = train_valid_df[train_valid_df['Class'] == 0]
#train_valid_df.head()

#Balancing the data by resampling from the smaller class group.
# df_class_0_over = df_class_0.sample(400, replace=True)
# train_valid_df = pd.concat([df_class_1, df_class_0_over], axis=0)
# print('Random over-sampling:')
# print(train_valid_df['Class'].value_counts())
# train_valid_df['Class'].value_counts().plot(kind='bar', title='Count (target)');
train_valid_df['Class'].value_counts().plot(kind='bar', title='Count (target)');
train_valid_df['Class'].value_counts()

# One thing to note is that it may be better to apply balancing during the cross-validation (https://www.researchgate.net/post/should_oversampling_be_done_before_or_within_cross-validation)
# This might mean that the above will lead to overfitting as we are creating too many duplicates of the 1s. Also note that because we split training and valid at the start, the sample size of each training set will change depending on the ratio of 1s and 0s we take from the original data for training. Thus the size of the train_df_bal will keep changing if we do it with this method.
#train_valid_df.shape

# Balancing the dataset results in the ID column reappearing in the dataset, so we will delete it again here.
# Remove ID as it does not give us any information.
# del train_valid_df['ID']
# from sklearn.utils import shuffle
# train_valid_df = shuffle(train_valid_df)

#### Why does imbalanced dataset lead to higher accuracy than balanced dataset?
# Imagine that your data is not easily separable. Your classifier isn't able to do a very good job at distinguishing between positive and negative examples, so it usually predicts the majority class for any example. In the unbalanced case, it will get 100 examples correct and 20 wrong, resulting in a 100/120 = 83% accuracy. But after balancing the classes, the best possible result is about 50%.

# The problem here is that accuracy is not a good measure of performance on unbalanced classes. It may be that your data is too difficult, or the capacity of your classifier is not strong enough. It's usually better to look at the confusion matrix to better understand how the classifier is working, or look at metrics other than accuracy such as the precision and recall, 𝐹1 score (which is just the harmonic mean of precision and recall), or AUC. These are typically all easy to use in common machine learning libraries like scikit-learn.

### Split Features and Target
#print("Dataset Length: ", len(train_valid_df))
#print("Dataset Shape: ", train_valid_df.shape)

# X contains all features and y contains target
X = train_valid_df.drop(columns=['Class'],axis=1)
y = train_valid_df['Class']
#X.shape
X.head()
y.head()

# Convert X into matrix format so cross validation function is satisfied.
X = X.values

## Balance Trial with SMOTE
# Convert y into matrix format so cross validation function is satisfied.
y = pd.DataFrame(y)
y = y.values.ravel()

# TRY BALANCE THE DATASET NOW USING SMOTE
# Found that imbalanced data was similar if not better (more consistent) than balanced dataset. 
# # Also for knn, the accuracy reduced a lot (down to 0.2).
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 33)
X, y = sm.fit_sample(X, y)
pd.Series(y).value_counts().plot.bar()

#### Perhaps don't need to balance? 
# You need to deal with imbalanced data set when the value of finding the minority class is much higher than that of finding the majority.

### 3.11 Feature engineering:
#### You may also come up with attributes derived from existing attributes. If this is the case, give an explanation of the new attributes that you have created. 
# I have already made dummy variables to make sure some of the columns which are categorical are turned into numerical values that the model can process!

# Featuretools is an open source library for performing automated feature engineering. Let us try doing this now: 
# import featuretools as ft
# es = ft.EntitySet(id = 'example')
# es = es.entity_from_dataframe(entity_id = 'example', dataframe = train_valid_df, index = 'ID')

### 3.12 Feature/Attribute selection: 
#### If applicable, clearly indicate which attributes you decide to remove in addition to those (obviously) irrelevant attributes that you have identified above and give a brief explanation why.

### 3.13 Data instances:
#### If you decide to make changes to the data instances with class labels (this may include selecting only a subset of the data, removing instances, randomizing/reordering instances, or synthetically injecting new data instances to the training data, etc. ), provide an explanation.
# Principal component analysis was performed to make sure there was no multicollinearity between variables. Dummy variables were also created prior to this to ensure that categorical variables could be read into the table.

### 3.14 PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
#df.shape
# There are currently 66 attributes (mainly from the dummy variables we created before).
#df.head()

# def knn_func(nn, listname):
#     model = KNeighborsClassifier(n_neighbors=nn)
#     model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for KNN: ", accuracy_valid)
#     listname.append(accuracy_valid)
# 
# def naivebayes_func(nb_list):
#     model = GaussianNB()
#     model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Naive Bayes: ", accuracy_valid)
#     nb_list.append(accuracy_valid)
#     
# def dtree_func(dtree_list):
#     model = DecisionTreeClassifier()
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Decision Tree: ", accuracy_valid)
#     dtree_list.append(accuracy_valid)
# 
# def random_forest_func(rf_list):
#     model = RandomForestClassifier(max_depth=2)
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for random forest: ", accuracy_valid)
#     rf_list.append(accuracy_valid)
#     
# def logreg_func(logreg_list):
#     model = LogisticRegression(solver='liblinear', multi_class='ovr')
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Logistic Regression: ", accuracy_valid)
#     logreg_list.append(accuracy_valid)
#     
# def lda_func(lda_list):
#     model = LinearDiscriminantAnalysis()
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for LDA: ", accuracy_valid, "\n")
#     lda_list.append(accuracy_valid)

from imblearn.ensemble import BalancedBaggingClassifier

def knn_func(nn, listname):
    model = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=nn),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for KNN: ", accuracy_valid)
    listname.append(accuracy_valid)

def naivebayes_func(nb_list):
    model = BalancedBaggingClassifier(base_estimator=GaussianNB(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Naive Bayes: ", accuracy_valid)
    nb_list.append(accuracy_valid)
    
def dtree_func(dtree_list):
    model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Decision Tree: ", accuracy_valid)
    dtree_list.append(accuracy_valid)

def random_forest_func(rf_list):
    model = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(max_depth=2, random_state=0),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Random Forest: ", accuracy_valid)
    rf_list.append(accuracy_valid)
    
def logreg_func(logreg_list):
    model = BalancedBaggingClassifier(base_estimator=LogisticRegression(solver='liblinear', multi_class='ovr'),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Logistic Regression: ", accuracy_valid)
    logreg_list.append(accuracy_valid)
    
def lda_func(lda_list):
    model = BalancedBaggingClassifier(base_estimator=LinearDiscriminantAnalysis(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for LDA: ", accuracy_valid, "\n")
    lda_list.append(accuracy_valid)


#Fitting the PCA algorithm with our Data
pca = PCA().fit(X)
#Plotting the Cumulative Summation of the Explained Variance
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Pulsar Dataset Explained Variance')
#plt.show()

pca = PCA(n_components=50, random_state=42)
'''Principal component analysis first applied without cross-validation to gage the impact it has. We will now
use the same formula and regenerate pca for each cross validation later on.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

rf = []
knn = []
nb = []
dtree = []
logreg = []
lda = []

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)
X_train = principalDf.values

principalComponents = pca.transform(X_valid)
principalDf = pd.DataFrame(data = principalComponents)
X_valid = principalDf.values

# model accuracy
#rf 
random_forest_func(rf)
# kNN
knn_func(7, knn)
# Naive Bayes
naivebayes_func(nb)
# Decision Tree
dtree_func(dtree)
# Logistic Regression
logreg_func(logreg)
# Linear Discriminant Analysis
lda_func(lda) 

### 3.15 RFE: Feature Extraction
# # Import your necessary dependencies
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# # Feature extraction
# model = LogisticRegression()
# rfe = RFE(model, 10)
# fit = rfe.fit(X, y)
# print("Num Features: %s" % (fit.n_features_))
# print("Selected Features: %s" % (fit.support_))
# print("Feature Ranking: %s" % (fit.ranking_))

# ## 3.16 Ridge regression
# First things first
# from sklearn.linear_model import Ridge
# ridge = Ridge(alpha=1.0)
# ridge.fit(X,y)
# Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#       normalize=False, random_state=None, solver='auto', tol=0.001)

# def pretty_print_coefs(coefs, names = None, sort = False):
#     if names == None:
#         names = ["X%s" % x for x in range(len(coefs))]
#     lst = zip(coefs, names)
#     if sort:
#         lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
#     return " + ".join("%s * %s" % (round(coef, 3), name)
#                                    for coef, name in lst)

# print ("Ridge model:", pretty_print_coefs(ridge.coef_))

### 3.2 KNN Trial & Error
# Let us try to determine the best n-neigbours to use for KNN via trial and error. We will start with 1 and then move up by 2.
# K in KNN is the number of instances that we take into account for determination of affinity with classes.

def knn_func(nn, listname):
    model = KNeighborsClassifier(n_neighbors=nn)
    model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    listname.append(accuracy_valid)

knn_1 = []
knn_3 = []
knn_5 = []
knn_7 = []
knn_9 = []
knn_11 = []
knn_13 = []
knn_15 = []
knn_17 = []

cv = StratifiedKFold(n_splits=10)
for train_index, valid_index in cv.split(X, y):
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    principalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data = principalComponents)
    X_train = principalDf.values

    principalComponents = pca.transform(X_valid)
    principalDf = pd.DataFrame(data = principalComponents)
    X_valid = principalDf.values
    
    knn_func(1, knn_1)
    knn_func(3, knn_3)
    knn_func(5, knn_5)
    knn_func(7, knn_7)
    knn_func(9, knn_9)
    knn_func(11, knn_11)
    knn_func(13, knn_13)
    knn_func(15, knn_15)
    knn_func(17, knn_17)

[[(np.var(knn_1)), (np.var(knn_3)), (np.var(knn_5)), (np.var(knn_7)), (np.var(knn_9)), (np.var(knn_11)), (np.var(knn_13)), (np.var(knn_15)), (np.var(knn_17))]]

[[sum(knn_1)/10, sum(knn_3)/10, sum(knn_5)/10, sum(knn_7)/10, sum(knn_9)/10, 
  sum(knn_11)/10, sum(knn_13)/10, sum(knn_15)/10, sum(knn_17)/10]]

# random_state = 42 Isn't that obvious? 
# 42 is the Answer to the Ultimate Question of Life, the Universe, and Everything.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)
X_train = principalDf.values

principalComponents = pca.transform(X_valid)
principalDf = pd.DataFrame(data = principalComponents)
X_valid = principalDf.values

# Credit: https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
k_range = range(1,100)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_valid)
    scores[k] = metrics.accuracy_score(y_valid, y_pred)
    scores_list.append(metrics.accuracy_score(y_valid, y_pred))

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# plot the relationships between K and the testing accuracy
#plt.plot(k_range, scores_list)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Testing accuracy')

# Another way to show this
# credit: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)
X_train = principalDf.values

principalComponents = pca.transform(X_valid)
principalDf = pd.DataFrame(data = principalComponents)
X_valid = principalDf.values

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_valid)
    error.append(np.mean(pred_i != y_valid))

#plt.figure(figsize=(12, 6))
#plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')
#plt.xlabel('K Value')
#plt.ylabel('Mean Error')

# Reference: https://stats.stackexchange.com/questions/151756/knn-1-nearest-neighbor

# Notice that the Mean error is higher near 0 which makes sense because this should not be the optimal nearest neighbour.

### With Bagging:
from imblearn.ensemble import BalancedBaggingClassifier
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)
X_train = principalDf.values

principalComponents = pca.transform(X_valid)
principalDf = pd.DataFrame(data = principalComponents)
X_valid = principalDf.values

error = []

# Calculating error for K values between 1 and 40

# I set replacement = True but prev it was set to False. Either way, I cannot see big diff?
for i in range(1, 80):
    model = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=i),
                                    sampling_strategy='auto',
                                    replacement=True,
                                    random_state=42)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_valid)
    error.append(np.mean(pred_i != y_valid))

#plt.figure(figsize=(12, 6))
#plt.plot(range(1, 80), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')
#plt.xlabel('K Value')
#plt.ylabel('Mean Error')

### 3.3 Cross validation:
#### How to evaluate the effectiveness of a classifier on the given data?
# Effectiveness of the classifier will be assessed by using the accuracy after cross-validation. Also looking for minimal variance between the accuracies and making sure the F-score is desirable.
#### How to address the issue of class imbalance in the training data?

# Already achieved previously. However because fixing the imbalance lead to more error, it was decided that data balancing would not be applied. This was both by oversampling and using the SMOTE data balancing package.
#### What is your choice of validation/cross-validation?

# 10-fold cross validation
#### For each classifier that you’ve selected, what is the validation/cross-validation performance? Give an interpretation of the confusion matrix.
#### For each classifier that you’ve selected, what is the estimated classification accuracy on the actual test data?

## Functions for all Classification Models with bagging
def knn_func(nn, listname):
    model = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=nn),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for KNN: ", accuracy_valid)
    listname.append(accuracy_valid)

def naivebayes_func(nb_list):
    model = BalancedBaggingClassifier(base_estimator=GaussianNB(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Naive Bayes: ", accuracy_valid)
    nb_list.append(accuracy_valid)
  
def dtree_func(dtree_list):
    model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Decision Tree: ", accuracy_valid)
    dtree_list.append(accuracy_valid)

def random_forest_func(rf_list):
    model = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(max_depth=2, random_state=0),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Decision Tree: ", accuracy_valid)
    rf_list.append(accuracy_valid)
  
def logreg_func(logreg_list):
    model = BalancedBaggingClassifier(base_estimator=LogisticRegression(solver='liblinear', multi_class='ovr'),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for Logistic Regression: ", accuracy_valid)
    logreg_list.append(accuracy_valid)
  
def lda_func(lda_list):
    model = BalancedBaggingClassifier(base_estimator=LinearDiscriminantAnalysis(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    accuracy_valid = accuracy_score(y_valid, predict_valid)
    #print("Accuracy for LDA: ", accuracy_valid, "\n")
    lda_list.append(accuracy_valid)

# def knn_func(nn, listname):
#     model = KNeighborsClassifier(n_neighbors=nn)
#     model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for KNN: ", accuracy_valid)
#     listname.append(accuracy_valid)
# 
# def naivebayes_func(nb_list):
#     model = GaussianNB()
#     model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Naive Bayes: ", accuracy_valid)
#     nb_list.append(accuracy_valid)
#     
# def dtree_func(dtree_list):
#     model = DecisionTreeClassifier()
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Decision Tree: ", accuracy_valid)
#     dtree_list.append(accuracy_valid)
# 
# def random_forest_func(rf_list):
#     model = RandomForestClassifier(max_depth=2, random_state=0)
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Decision Tree: ", accuracy_valid)
#     rf_list.append(accuracy_valid)
#     
# def logreg_func(logreg_list):
#     model = LogisticRegression(solver='liblinear', multi_class='ovr')
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for Logistic Regression: ", accuracy_valid)
#     logreg_list.append(accuracy_valid)
#     
# def lda_func(lda_list):
#     model = LinearDiscriminantAnalysis()
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     accuracy_valid = accuracy_score(y_valid, predict_valid)
#     #print("Accuracy for LDA: ", accuracy_valid, "\n")
#     lda_list.append(accuracy_valid)

# cross_validation.Bootstrap is deprecated. cross_validation.KFold or cross_validation.ShuffleSplit are recommended instead.
print("Applying classification models...")
knn = []
nb = []
dtree = []
logreg = []
lda = []
svm = []
rf = []

'''First I trialed the normal K fold cross validation, but then realised the stratified k-fold cross validation
worked much better with a more consistent accuracy across trials.'''

# cv = KFold(n_splits=10, random_state=20, shuffle=False)
# for train_index, valid_index in cv.split(X):
cv = StratifiedKFold(n_splits=10)
for train_index, valid_index in cv.split(X, y):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", valid_index)
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
    
    principalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data = principalComponents)
    X_train = principalDf.values

    principalComponents = pca.transform(X_valid)
    principalDf = pd.DataFrame(data = principalComponents)
    X_valid = principalDf.values

    # kNN
    knn_func(7, knn)   
    # Naive Bayes
    naivebayes_func(nb)
    # Decision Tree
    dtree_func(dtree)
    # Logistic Regression
    logreg_func(logreg)
    # Linear Discriminant Analysis
    lda_func(lda) 
    # Random Forest
    random_forest_func(rf)

knn
nb
dtree
logreg
lda
rf

# Averages for each model
print("\nOverall Accuracy for KNN with 10-fold cross-validation:", sum(knn)/10)
print("\nOverall Accuracy for Naive Bayes with 10-fold cross-validation:", sum(nb)/10)
print("\nOverall Accuracy for Decision tree with 10-fold cross-validation:", sum(dtree)/10)
print("\nOverall Accuracy for Logistic Reg with 10-fold cross-validation:", sum(logreg)/10)
print("\nOverall Accuracy for LDA with 10-fold cross-validation:", sum(lda)/10)
print("\nOverall Accuracy for Random Forest with 10-fold cross-validation:", sum(rf)/10)
### 3.31 Variance of Accuracy
print("\nVariance of the Accuracies: ")
print("Variance for KNN Accuracy: ", np.var(knn))
print("Variance for NB Accuracy: ", np.var(nb))
print("Variance for Decision Tree Accuracy: ", np.var(dtree))
print("Variance for Logistic Regression Accuracy: ", np.var(logreg))
print("Variance for LDA Accuracy: ", np.var(lda))
print("Variance for Random Forest Accuracy: ", np.var(rf))

### 3.32 F-Score:
## Functions to predict valid class and test class for all Classification Models
'''In order to make the prediction even more accurate, we will use the whole 1000 instances to predict this time
in other words, we will not have a validation set because we have already verified that the model works beforehand.'''

# def knn_valid(nn):
#     model = KNeighborsClassifier(n_neighbors=nn)
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     return predict_valid
#     
# def naivebayes_valid():
#     model = GaussianNB()
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     return predict_valid
#     
# def dtree_valid():
#     model = DecisionTreeClassifier()
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     return predict_valid
# 
# def logreg_valid():
#     model = LogisticRegression(solver='liblinear', multi_class='ovr')
#     model = model.fit(X_train, y_train)
#     predict_valid = model.predict(X_valid)
#     return predict_valid
#     
# def lda_valid():
#     model = LinearDiscriminantAnalysis()
#     model = model.fit(X_train,y_train)
#     predict_valid = model.predict(X_valid)
#     return predict_valid

def knn_valid(nn):
    model = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=nn),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid
    
def naivebayes_valid():
    model = BalancedBaggingClassifier(base_estimator=GaussianNB(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid
    
def dtree_valid():
    model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid

def random_forest_valid():
    model = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(max_depth=2, random_state=0),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid

    
def logreg_valid():
    model = BalancedBaggingClassifier(base_estimator=LogisticRegression(solver='liblinear', multi_class='ovr'),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid
    
def lda_valid():
    model = BalancedBaggingClassifier(base_estimator=LinearDiscriminantAnalysis(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_valid = model.predict(X_valid)
    return predict_valid


# kNN
knn_pred_valid = knn_valid(7)
# Naive Bayes
nb_pred_valid = naivebayes_valid()
# Decision Tree
dtree_pred_valid = dtree_valid()
# Logistic Regression
logreg_pred_valid = logreg_valid()
# Linear Discriminant Analysis
lda_pred_valid = lda_valid()
#Random Forest
rf_pred_valid = random_forest_valid()

knn_pred_valid
nb_pred_valid
dtree_pred_valid
logreg_pred_valid
lda_pred_valid 
rf_pred_valid
y_valid

from sklearn.metrics import classification_report, confusion_matrix

cmtx = pd.DataFrame(
    confusion_matrix(y_valid, knn_pred_valid), 
    index=['true: 0', 'true: 1'], 
    columns=['pred: 0', 'pred: 1']
)
print("\nKNN Confusion Matrix:")
print(cmtx)

cmtx = pd.DataFrame(
    confusion_matrix(y_valid, nb_pred_valid), 
    index=['true: 0', 'true: 1'], 
    columns=['pred: 0', 'pred: 1']
)
print("\nNaive Bayes Confusion Matrix:")
print(cmtx)

cmtx = pd.DataFrame(
    confusion_matrix(y_valid, dtree_pred_valid), 
    index=['true: 0', 'true: 1'], 
    columns=['pred: 0', 'pred: 1']
)
print("\nDecision Tree Confusion Matrix:")
print(cmtx)

cmtx = pd.DataFrame(
    confusion_matrix(y_valid, logreg_pred_valid), 
    index=['true: 0', 'true: 1'], 
    columns=['pred: 0', 'pred: 1']
)
print("\nLogistic Regression Confusion Matrix")
print(cmtx)

cmtx = pd.DataFrame(
    confusion_matrix(y_valid, lda_pred_valid), 
    index=['true: 0', 'true: 1'], 
    columns=['pred: 0', 'pred: 1']
)
print("\nLDA Confusion Matrix:")
print(cmtx)

print("\nRandom Forest Confusion Matrix")
cmtx = pd.DataFrame(
    confusion_matrix(y_valid, rf_pred_valid),
    index=['true: 0', 'true: 1'],
    columns=['pred: 0', 'pred: 1']
)
print(cmtx)

print("\nClassification Report for KNN:")
print(classification_report(y_valid, knn_pred_valid))
print("\nClassification Report for Naive Bayes:")
print(classification_report(y_valid, nb_pred_valid))
print("\nClassification Report for Decision Tree:")
print(classification_report(y_valid, dtree_pred_valid))
print("\nClassification Report for Log Regression:")
print(classification_report(y_valid, logreg_pred_valid))
print("\nClassification Report for LDA:")
print(classification_report(y_valid, lda_pred_valid))
print("\nClassification Report for Random Forest:")
print(classification_report(y_valid, rf_pred_valid))

from sklearn.metrics import f1_score
print("\nF-score for KNN:")
print(f1_score(knn_pred_valid, y_valid, average='micro'))

print("\nF-score for Naive Bayes:")
print(f1_score(nb_pred_valid, y_valid, average='micro'))

print("\nF-score for Decision Tree:")
print(f1_score(dtree_pred_valid, y_valid, average='micro'))

print("\nF-score for Logistic Regression:")
print(f1_score(logreg_pred_valid, y_valid, average='micro'))

print("\nF-score for LDA:")
print(f1_score(lda_pred_valid, y_valid, average='micro'))

print("\nF-score for Random Forest:")
print(f1_score(rf_pred_valid, y_valid, average='micro'))

# The F score, also called the F1 score or F measure, is a measure of a test’s accuracy. The F score is defined as the weighted harmonic mean of the test’s precision and recall. Precision, also called the positive predictive value, is the proportion of positive results that truly are positive. Recall, also called sensitivity, is the ability of a test to correctly identify positive results to get the true positive rate. The F score reaches the best value, meaning perfect precision and recall, at a value of 1. The worst F score, which means lowest precision and lowest recall, would be a value of 0. 

# Given that its cancer data (which inherently has class imbalance), F1 score (not accuracy) will be the correct metric to use. Here's wiki for more on F1 score. It basically lets you chose how you want to relatively weigh Precision and Recall.
# Small changes to K in training set can lead to large changes in decision boundary so you'll need to cross validate and see if the results generalize well.

### 3.4 Classifier comparison:

#### Compare the classification performance between difference classifiers. You need to select at least two (2) evaluation metrics, for example F-measure and classification accuracy, when comparing them. Your comparison must take into account the variation between different runs due to cross-validation.
# - F-measure
# - Classification accuracy
# LDA and logistic regression seem to perform the best in terms of accuracy (approx 77%) and F-score (approx 0.81).

#### Based on the comparison, select the best two (2) classification schemes for final prediction. Note that the two classification schemes can be one type of classifier, but with two different parameters. Clearly indicate the final choice of parameters if they are not the default values.

### 3.5 Prediction:
#### Use the best two classification schemes that you have identified in the previous step to predict the missing class labels of the last 100 samples in the original data set.
test_df = pd.read_csv('test_set.csv')
test_df.head()

# X contains all features and y contains target
X_test = test_df.drop(columns=['Class'],axis=1)
y_test = test_df['Class']
X_test.head()

'''We are now going to use both the training and validation data in order to make the optimal model for the test data, so we are reloading this dataframe and reproducing the pca analysis. 
This should theoretically make the model better at predicting the test data. Note that X contains all features 
and y contains target'''

X_train = train_valid_df.drop(columns=['Class'],axis=1)
y_train = train_valid_df['Class']
X_train.head()
X.shape

'''Note that pca.fit_transform was uses in the first PCA but in the test pca, 
pca.transform is used to maintain the same space for the pca (same mean and st dev).'''
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents)
X_train = principalDf.values

principalComponents = pca.transform(X_test)
principalDf = pd.DataFrame(data = principalComponents)
X_test = principalDf.values

## Functions to predict valid class and test class for all Classification Models
'''In order to make the prediction even more accurate, we will use the whole 1000 instances to predict this time
in other words, we will not have a validation set because we have already verified that the model works beforehand.'''

# def knn_test(nn):
#     model = KNeighborsClassifier(n_neighbors=nn)
#     model = model.fit(X_train, y_train)
#     predict_test = model.predict(X_test)
#     return predict_test
#     
# def naivebayes_test():
#     model = GaussianNB()
#     model = model.fit(X_train, y_train)
#     predict_test = model.predict(X_test)
#     return predict_test
#     
# def dtree_test():
#     model = DecisionTreeClassifier()
#     model = model.fit(X_train, y_train)
#     predict_test = model.predict(X_test)
#     return predict_test
#     
# def logreg_test():
#     model = LogisticRegression(solver='liblinear', multi_class='ovr')
#     model = model.fit(X_train, y_train)
#     predict_test = model.predict(X_test)
#     return predict_test
#     
# def lda_test():
#     model = LinearDiscriminantAnalysis()
#     model = model.fit(X_train,y_train)
#     predict_test = model.predict(X_test)
#     return predict_test

def knn_test(nn):
    model = BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=nn),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    return predict_test
    
def naivebayes_test():
    model = BalancedBaggingClassifier(base_estimator=GaussianNB(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    return predict_test
    
def dtree_test():
    model = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    return predict_test

def random_forest_test():
    model = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(max_depth=2, random_state=0),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    return predict_test

    
def logreg_test():
    model = BalancedBaggingClassifier(base_estimator=LogisticRegression(solver='liblinear', multi_class='ovr'),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train, y_train)
    predict_test = model.predict(X_test)
    return predict_test
    
def lda_test():
    model = BalancedBaggingClassifier(base_estimator=LinearDiscriminantAnalysis(),
            sampling_strategy='auto',
            replacement=False,
            random_state=0)
    model = model.fit(X_train,y_train)
    predict_test = model.predict(X_test)
    return predict_test

# kNN
knn_pred_test = knn_test(7)
# Naive Bayes
nb_pred_test = naivebayes_test()
# Decision Tree
dtree_pred_test = dtree_test()
# Logistic Regression
logreg_pred_test = logreg_test()
# Linear Discriminant Analysis
lda_pred_test = lda_test() 
# Random Forest
rf_pred_test = random_forest_test()

knn_pred_test
nb_pred_test
dtree_pred_test
logreg_pred_test
lda_pred_test
rf_pred_test

print("\n\nNow let's do a 50% Data Distribution Check for the predicted test class: ")
print("Number of 1 in predicted test class for Logistic Reg: ", logreg_pred_test.sum())
print("Number of 1 in predicted test class for Random Forest: ", rf_pred_test.sum())
print("\nProportion is as expected as there are 50/50 1 and 0 in the actual test class.")


# Because logistic regression and lda had the highest accuracy when looking at the validation we will use these two models to include in our csv file.

# ### Produce a CSV file with the name predict.csv that contain your prediction in a similar format: the first column is the sample ID, the second and third columns are the predicted class labels. This file must be submitted electronically with the electronic copy of the report via Blackboard. An example of such a file is given below.
# - IMPORTANT: Please ensure that your prediction is correctly formatted as required. Your marks will be deduced if your prediction file does not meet the above requirements. If your submitted file has more than 2 predictions, only the first two will be marked. No correction to the prediction is allowed after your assignment is submitted.
# -  You must also indicate clearly in the report your estimated prediction accuracy. This should be based on the validation study.

id_list = []
for i in range(1001, 1101):
    id_list.append(i)

import csv
with open('predict.csv', 'w', newline='') as f_output:         
    csv_output = csv.writer(f_output)
    csv_output.writerow(["ID", "logReg", "randomForest"])
    csv_output.writerows(zip(id_list, logreg_pred_test, rf_pred_test))
