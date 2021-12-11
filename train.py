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
import json
from sklearn.inspection import permutation_importance
from sklearn import svm
from sklearn.model_selection import GridSearchCV

################################################
#train a random forest model as a baseline model
################################################
#read data
df = pd.read_csv('processed_data.csv')
df_x = df.drop('Label', axis=1)
y = df['Label']

#create svm model
clf = svm.SVC()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
n_scores = cross_val_score(clf, df_x, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#write score to a file
#with open("metrics.json", 'w') as outfile:
#        json.dump({"Balanced Accuracy (mean recall)":np.mean(n_scores), "Standard Deviation":np.std(n_scores)}, outfile)

########################################
#feature importance and confusion matrix
########################################
#use label encoder to convert the categorical labels into numerical ones
le = LabelEncoder()
y = le.fit_transform(y)

#split the dataset into train and test set and tdo a grid search to tune the SVM hyperparameters. 
X_train, X_test, y_train, y_test = train_test_split(df_x, y, test_size = 0.2, random_state = 42)

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, grid_predictions))

#calculate sensitivity and specificity for multi-class classification
cnf_matrix = metrics.confusion_matrix(y_test, grid_predictions)
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

#write score to a file
with open("metrics.json", 'w') as outfile:
        json.dump({"Balanced Accuracy (mean recall)":metrics.balanced_accuracy_score(y_test, grid_predictions), "Specificity":np.mean(TNR)}, outfile)

#plot the most important features for the model
perm_importance = permutation_importance(grid, X_test, y_test)
feature_names = X_train.columns
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(8, 12), dpi=80)
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()

#plot confusion matrix
c_mat = pd.DataFrame(data=np.column_stack((le.inverse_transform(y_test),le.inverse_transform(grid_predictions))), columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(c_mat['y_Actual'], c_mat['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=120)
