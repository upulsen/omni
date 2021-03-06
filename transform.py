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
#df.drop(columns=['Subject_ID'], inplace=True)

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
y = df[['Subject_ID','Label']]

#identify the numerical colums to do mean imputation for missing values
numerical_columns = df_x.select_dtypes(include='number').columns 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_x[numerical_columns] = imputer.fit_transform(df_x[numerical_columns])

#find categorical columns and use one hot encoding to conver them to numerical columns
cat_columns = df_x.select_dtypes(exclude='number').columns
df_x = pd.get_dummies(df_x, columns=cat_columns, drop_first=True)

#write processed data
wr = pd.merge(df_x, y, on='Subject_ID', how="inner")
wr.to_csv('processed_data.csv')
