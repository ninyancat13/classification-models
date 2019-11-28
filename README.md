# classification-models

## 1.1 Major Findings
### 1.11 Data Preparation
o Att14 and att17 are irrelevant because whether they are included in the analysis or not will not make a difference to the outcome.\
o Att3 and att9 are categorical attributes with very few missing values so these missing values were imputed by using the mode. Att25 and att28 however are numerical with a few missing values so these were imputed using the mean. Att13 and att19 had a very large number of missing values so these attributes were removed.\
o Att8 and att24 are duplicate columns so att24 was removed as this would be redundant information for the classification. There were no duplicate rows.\
o Att1 to att12 which had alphabetical labels were changed into categorical datatype by engineering dummy variables. The other categorical variables which had numerical values were simply converted into categorical datatype without producing dummy variables.\
o Att18, att25 and att28 are all skewed to the right. Thus, a log transformation was applied.\
o Training and validation data (1000 instances) were split off from the test data (100 instances).\
o Standardisation was applied to the training/valid set and then also applied to the test set with
the same mean.\
o Multicollinearity check was applied to the dataset and there was no multicollinearity present.
This means LDA can now be conducted.\
o Principal component analysis was applied to the dataset. From 65 attributes, 50 principal
components were produced.\
o Data balancing was applied using SMOTE, balanced bagging classifier and stratified cross
validation (stratified cross validation ensured balanced distribution both for training and validation sets).

### 1.12 Classification
o Hyperparameters were tuned for all models, including KNN, Naïve Bayes, Decision Tree, Logistic Regression, Linear Discriminant Analysis (LDA) and Random Forest. Some hyperparameters which were adjusted include K in KNN, max depth in random forest and decision trees.\
o Stratified 10-fold cross validation was used to evaluate the effectiveness of the classifier. o Average accuracy was derived as well as the variances for each model. F-scores were also produced alongside confusion matrices for each model using scikit learn.\
o Overall looking at recall and precision it can be seen that recall is much higher for all the models compared to precision. This signifies that most models are more likely to predict a 1 as 0 than it is to predict a 0 as a 1.\
o Logistic Regression (with accuracy of 0.761, variance of 0.001393 and F-score of 0.788) and
Random Forest (with accuracy of 0.766, variance of 0.001796 and F-score of 0.795) were chosen as the best two classification schemes because of their high accuracy, lower variances and higher F-scores.


## 1.2 Lessons Learned 
### 1.21 Data Preparation
o Sometimes without domain knowledge and context of a dataset, it is hard to make decisions in the pre-processing step.\
o Deciding when to split the training, validation and test is an important decision. In the current dataset, there were very few missing values to impute, so it was not necessary to split the test from the training data prior to imputation to prevent data leakage. When to split the dataset is a decision that needs to be made depending on the type of dataset that is analysed.\
o Dummy variables, although not always necessary, are a good way to change numerical data to categorical format (binary values) and feature engineer new columns. This may also improve the PCA because each level of a factor may influence the model with different weightage.\
o Data pre-processing is arguably one of the most important skills for classification and the decisions that are made will affect the outcome of the classification models heavily.

### 1.22 Classification
o Balancing dataset is very important since training on an unbalanced dataset will mean it will automatically predict the majority class more often than the minority and result in bias in the model.\
o Initial accuracies on oversampled dataset were too high. Overfitting was found to be the cause of this high accuracy. Overfitting can result from balancing data (resampling from the minority class) due to duplicates. SMOTE and balanced bagging classifiers as well as stratified sampling in cross validation is one way to prevent this problem.\
o Important to tune hyperparameters manually in classification modelling as it cannot be pre- determined. It can only be selected based on classifier outcomes such as with accuracy or the F-score.\
o When finding the optimal K for KNN, it is necessary to find the error rate that is small enough so that only the samples which are relevant are present but also big enough that there is no overfitting. Determining the value of K in KNN is a difficult problem. K = 1 showed the least mean error but this is most likely due to overfitting. It is difficult to access which K value is optimal just looking at the accuracy of the model or mean error rates.\
o Important to use whole training dataset (1000 instances) to train the final model and predict the test class labels.\
o Weka is a powerful tool because it allows better insight into hyperparameter tuning. Thus, even though python is very powerful, it probably is easier to tune hyperparameters quickly on Weka to get some insight.
