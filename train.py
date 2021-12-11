#################
#import libraries
#################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#######################################
#read data and transform to fit a model
#######################################
df = pd.read_csv('o8t_testdata.csv')

# drop columns with more than 80% null values
df.dropna(thresh = df.shape[0]*0.2, how = 'all', axis = 1, inplace = True)
df.drop(columns=['Subject_ID'], inplace=True)

#drop any row that has a null value in Label column
df.dropna(subset=['Label'], inplace=True)

#drop the rows that has MCI Dementia and MCI naMCI_MD
df = df[~df['Label'].isin(['MCI Dementia', 'MCI naMCI_MD'])]

#change dtypes and drop C7-STT_B
df['C7-STT_A'].replace(['Unable to complete'], '', inplace=True)
df['C7-STT_B'].replace(['Unable to complete'], '', inplace=True)
df[['C7-STT_A', 'C7-STT_B']] = df[['C7-STT_A', 'C7-STT_B']].apply(pd.to_numeric)
df.drop(columns=['C7-STT_B'], inplace=True)

#separate the dataset into data and labels
df_x = df.drop('Label', axis=1)
y = df['Label']

#identify the numerical colums to do mean imputation for missing values
numerical_columns = df_x.select_dtypes(include='number').columns 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_x[numerical_columns] = imputer.fit_transform(df_x[numerical_columns])

#find categorical columns and use one hot encoding to conver them to numerical columns
cat_columns = df_x.select_dtypes(exclude='number').columns
df_x = pd.get_dummies(df_x, columns=cat_columns, drop_first=True)

################################################
#train a random forest model as a baseline model
################################################
clf = RandomForestClassifier(max_depth=10, n_estimators=50, random_state=0)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
n_scores = cross_val_score(clf, df_x, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#write score to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write('Balanced Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

########################################
#feature importance and confusion matrix
########################################
#use label encoder to convert the categorical labels into numerical ones
le = LabelEncoder()
y = le.fit_transform(y)

#split the dataset into train and test set and train a random forest classifier on the train set. 
X_train, X_test, y_train, y_test = train_test_split(df_x, y, test_size = 0.2, random_state = 42)
clf2 = RandomForestClassifier(max_depth=10, random_state=0)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))

#write score to a file
with open("metrics.txt", 'a') as outfile:
        outfile.write('\nBalanced Accuracy for train/test split: %.3f' % metrics.balanced_accuracy_score(y_test, y_pred))

#plot the most important features for the model
feature_names = X_train.columns
plt.figure(figsize=(8, 12), dpi=120)
sorted_idx = clf2.feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], clf2.feature_importances_[sorted_idx])
plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()

#plot confusion matrix
c_mat = pd.DataFrame(data=np.column_stack((le.inverse_transform(y_test),le.inverse_transform(y_pred))), columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(c_mat['y_Actual'], c_mat['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=120)
