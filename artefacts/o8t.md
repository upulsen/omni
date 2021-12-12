```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
```


```python
# import data
df = pd.read_csv('o8t_testdata.csv')
```


```python
pd.set_option('display.max_columns', None)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Subject_ID</th>
      <th>Label</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>History of cerebrovascular disease</th>
      <th>History of hypertension</th>
      <th>History of diabetes</th>
      <th>History of coronary heart disease</th>
      <th>History of hyperlipidemia</th>
      <th>History of anemia</th>
      <th>History of CO poisoning</th>
      <th>History of general anesthesia during surgery</th>
      <th>History of abnormal thyroid function</th>
      <th>History of traumatic brain injury</th>
      <th>Family history of dementia</th>
      <th>Smoking history</th>
      <th>Drinking history</th>
      <th>Unnamed: 21</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C7-STT_B</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>MCI</td>
      <td>Male</td>
      <td>77</td>
      <td>165.0</td>
      <td>90.0</td>
      <td>9</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>20.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>0</td>
      <td>218</td>
      <td>30.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Dementia</td>
      <td>Male</td>
      <td>81</td>
      <td>169.0</td>
      <td>70.0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>8.0</td>
      <td>Unable to complete</td>
      <td>Unable to complete</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Normal</td>
      <td>Female</td>
      <td>77</td>
      <td>155.0</td>
      <td>55.0</td>
      <td>12</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>26.0</td>
      <td>16.0</td>
      <td>68</td>
      <td>150</td>
      <td>36.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Normal</td>
      <td>Male</td>
      <td>75</td>
      <td>169.0</td>
      <td>75.0</td>
      <td>9</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Normal</td>
      <td>Male</td>
      <td>68</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>24.0</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>27.0</td>
      <td>15.0</td>
      <td>34</td>
      <td>94</td>
      <td>36.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Looking at this subset, we can clearly see there are missing values and sentinel values to signify missing values such as 'Unable to complete'. Column 'Unnamed: 21 seems empty which we will check and remove. 


```python
# drop columns with more than 80% null values
df.dropna(thresh = df.shape[0]*0.2, how = 'all', axis = 1, inplace = True)
df.drop(columns=['Subject_ID'], inplace=True)
```

I decided to drop any column with more than 80% null values as that will not add any value to the model. I'm also dropping 'Subject ID' column which is a subject identifier and will not add value to a ML model.


```python
df.dtypes
```




    Label                                                   object
    Gender                                                  object
    Age                                                      int64
    Education                                              float64
    Height                                                 float64
    Weight                                                   int64
    History of cerebrovascular disease                      object
    History of hypertension                                 object
    History of diabetes                                     object
    History of coronary heart disease                       object
    History of hyperlipidemia                               object
    History of anemia                                       object
    History of CO poisoning                                 object
    History of general anesthesia during surgery            object
    History of abnormal thyroid function                    object
    History of traumatic brain injury                       object
    Family history of dementia                              object
    Smoking history                                         object
    Drinking history                                        object
    NPI                                                    float64
    MoCAB                                                  float64
    MMSE                                                   float64
    IADL                                                   float64
    HAMA                                                   float64
    HAMD                                                   float64
    C1 HVLT(immediate memory)                              float64
    C5 HVLT delayed recall 5min                            float64
    C8 HVLT delayed recall 20min                           float64
    C4 logical memory WMS                                  float64
    C6 Boston Naming Test                                  float64
    C3 articulateness and verbal fluencey-vegetable BNT    float64
    C7-STT_A                                                object
    C7-STT_B                                                object
    C2 CFT Rey-limitation                                  float64
    C9 CFT Rey-recall                                      float64
    HD1 depressive mood                                    float64
    HD2 guilty                                             float64
    HD3 suicidal                                           float64
    HD7 work & interests                                   float64
    HA6 Total Score of Depressive Mood \n                  float64
    Total score of Depression core factors                 float64
    HA1 Anxiety                                            float64
    HA2 Tension                                            float64
    HA3 Fear                                               float64
    HA14 Interview perfomance                              float64
    Total score of Anxiety factors                         float64
    dtype: object



This has a mix of numerical and categorical columns. Label column will be used to build the classification model. Let us look at the distribution of labels.


```python
df['Label'].hist()
```




    <AxesSubplot:>




    
![png](output_8_1.png)
    


It appears that this is a multi-class classification problem and the classes of interest are 'Normal', 'MCI', and 'Dementia'. I am going to disregard 'MCI Dementia' and 'MCI naMCI_MD' classes as they only have one instance each which will not be enough to build a model. We can also note that distribution of MCI and Normal patients are fairly similar in numbers whereas patients with Dementia are less than half. Due to this imbalance, we will need to pay particular attention to the performance metrics we'll be using. 


```python
#drop any row that has a null value in Label column
df.dropna(subset=['Label'], inplace=True)
#drop the rows that has MCI Dementia and MCI naMCI_MD
df = df[~df['Label'].isin(['MCI Dementia', 'MCI naMCI_MD'])]
```

We will look at the distribution of demographic variables such as Age, Education and Gender with respect to patient diagnosis. It appears none of these carry significant differentiating power. 


```python
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,6))

#plot distribution of Age grouped by the diagnosis
df['Age'].hist(by=df['Label'], range=[50, 100], align='mid', ax=axes)
plt.suptitle('Age Distribution', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Age', ha='center')
fig.text(0.04, 0.5, 'count', va='center', rotation='vertical')
plt.savefig('AgeDistro.png', bbox_inches='tight')
```


    
![png](output_12_0.png)
    



```python
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,6))

#plot distribution of Education grouped by the diagnosis
df['Education'].hist(by=df['Label'], range=[140, 180], align='mid', ax=axes)
plt.suptitle('Education Distribution', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Education', ha='center')
fig.text(0.04, 0.5, 'count', va='center', rotation='vertical')
plt.savefig('EduDistro.png', bbox_inches='tight')
```


    
![png](output_13_0.png)
    



```python
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12,6))

#plot distribution of Education grouped by the diagnosis
df['Gender'].hist(by=df['Label'], align='mid', ax=axes)
plt.suptitle('Education Distribution', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Gender', ha='center')
fig.text(0.04, 0.5, 'count', va='center', rotation='vertical')
plt.savefig('GenderDistro.png', bbox_inches='tight')
```


    
![png](output_14_0.png)
    



```python
#are there any null values?
df.isnull().sum()
```




    Label                                                  0
    Gender                                                 0
    Age                                                    0
    Education                                              2
    Height                                                 2
    Weight                                                 0
    History of cerebrovascular disease                     0
    History of hypertension                                0
    History of diabetes                                    0
    History of coronary heart disease                      0
    History of hyperlipidemia                              0
    History of anemia                                      0
    History of CO poisoning                                0
    History of general anesthesia during surgery           0
    History of abnormal thyroid function                   0
    History of traumatic brain injury                      0
    Family history of dementia                             0
    Smoking history                                        0
    Drinking history                                       0
    NPI                                                    2
    MoCAB                                                  2
    MMSE                                                   2
    IADL                                                   2
    HAMA                                                   2
    HAMD                                                   1
    C1 HVLT(immediate memory)                              4
    C5 HVLT delayed recall 5min                            4
    C8 HVLT delayed recall 20min                           4
    C4 logical memory WMS                                  4
    C6 Boston Naming Test                                  4
    C3 articulateness and verbal fluencey-vegetable BNT    4
    C7-STT_A                                               4
    C7-STT_B                                               4
    C2 CFT Rey-limitation                                  4
    C9 CFT Rey-recall                                      4
    HD1 depressive mood                                    1
    HD2 guilty                                             1
    HD3 suicidal                                           1
    HD7 work & interests                                   1
    HA6 Total Score of Depressive Mood \n                  1
    Total score of Depression core factors                 1
    HA1 Anxiety                                            2
    HA2 Tension                                            2
    HA3 Fear                                               2
    HA14 Interview perfomance                              2
    Total score of Anxiety factors                         2
    dtype: int64



It appears that there are no null values in the categorical columns. For null values in numerical columns, I'm going to use mean imputation which will replace the null values by the mean of that column. Before doing that though, I am going to replace sentinel values in C7-STT_A and C7-STT_B and convert it to a numerical column. 


```python
df['C7-STT_A'].replace(['Unable to complete'], '', inplace=True)
df['C7-STT_B'].replace(['Unable to complete'], '', inplace=True)
df[['C7-STT_A', 'C7-STT_B']] = df[['C7-STT_A', 'C7-STT_B']].apply(pd.to_numeric)
```


```python
df.isnull().sum()
```




    Label                                                   0
    Gender                                                  0
    Age                                                     0
    Education                                               2
    Height                                                  2
    Weight                                                  0
    History of cerebrovascular disease                      0
    History of hypertension                                 0
    History of diabetes                                     0
    History of coronary heart disease                       0
    History of hyperlipidemia                               0
    History of anemia                                       0
    History of CO poisoning                                 0
    History of general anesthesia during surgery            0
    History of abnormal thyroid function                    0
    History of traumatic brain injury                       0
    Family history of dementia                              0
    Smoking history                                         0
    Drinking history                                        0
    NPI                                                     2
    MoCAB                                                   2
    MMSE                                                    2
    IADL                                                    2
    HAMA                                                    2
    HAMD                                                    1
    C1 HVLT(immediate memory)                               4
    C5 HVLT delayed recall 5min                             4
    C8 HVLT delayed recall 20min                            4
    C4 logical memory WMS                                   4
    C6 Boston Naming Test                                   4
    C3 articulateness and verbal fluencey-vegetable BNT     4
    C7-STT_A                                               19
    C7-STT_B                                               44
    C2 CFT Rey-limitation                                   4
    C9 CFT Rey-recall                                       4
    HD1 depressive mood                                     1
    HD2 guilty                                              1
    HD3 suicidal                                            1
    HD7 work & interests                                    1
    HA6 Total Score of Depressive Mood \n                   1
    Total score of Depression core factors                  1
    HA1 Anxiety                                             2
    HA2 Tension                                             2
    HA3 Fear                                                2
    HA14 Interview perfomance                               2
    Total score of Anxiety factors                          2
    dtype: int64



It appears that more than 20% of C7-STT_B is missing, while we could do mean imputation, this may add an inadvertent bias, especially considering the low sample size. Therefore, I will remove C7-STT_B but retain C7-STT_A after mean imputation. STT_A and STT_B evaluate the executive functions so retaining at least one is warranted despite the high number of missing values on C7-STT_A as well.


```python
df.drop(columns=['C7-STT_B'], inplace=True)
```


```python
#separate the dataset into data and labels
df_x = df.drop('Label', axis=1)
y = df['Label']

#identify the numerical colums to do mean imputation for missing values
numerical_columns = df_x.select_dtypes(include='number').columns 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_x[numerical_columns] = imputer.fit_transform(df_x[numerical_columns])
```


```python
df_x.isnull().sum()
```




    Gender                                                 0
    Age                                                    0
    Education                                              0
    Height                                                 0
    Weight                                                 0
    History of cerebrovascular disease                     0
    History of hypertension                                0
    History of diabetes                                    0
    History of coronary heart disease                      0
    History of hyperlipidemia                              0
    History of anemia                                      0
    History of CO poisoning                                0
    History of general anesthesia during surgery           0
    History of abnormal thyroid function                   0
    History of traumatic brain injury                      0
    Family history of dementia                             0
    Smoking history                                        0
    Drinking history                                       0
    NPI                                                    0
    MoCAB                                                  0
    MMSE                                                   0
    IADL                                                   0
    HAMA                                                   0
    HAMD                                                   0
    C1 HVLT(immediate memory)                              0
    C5 HVLT delayed recall 5min                            0
    C8 HVLT delayed recall 20min                           0
    C4 logical memory WMS                                  0
    C6 Boston Naming Test                                  0
    C3 articulateness and verbal fluencey-vegetable BNT    0
    C7-STT_A                                               0
    C2 CFT Rey-limitation                                  0
    C9 CFT Rey-recall                                      0
    HD1 depressive mood                                    0
    HD2 guilty                                             0
    HD3 suicidal                                           0
    HD7 work & interests                                   0
    HA6 Total Score of Depressive Mood \n                  0
    Total score of Depression core factors                 0
    HA1 Anxiety                                            0
    HA2 Tension                                            0
    HA3 Fear                                               0
    HA14 Interview perfomance                              0
    Total score of Anxiety factors                         0
    dtype: int64



As can be seen, there are no missing values in the dataset now. Finally, we will explore the summary statistics of the dataset and will look at highly correlated variables in the dataset. It appears some variables are highly correlated, for instance MMSE and NPI. We may able to refrain from conducting the highly correlated tests if needed and will further look into this by exploring the importance each of these variables have to make an automated diagnosis.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>210.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>210.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>209.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>191.000000</td>
      <td>206.000000</td>
      <td>206.000000</td>
      <td>209.000000</td>
      <td>209.000000</td>
      <td>209.000000</td>
      <td>209.000000</td>
      <td>209.000000</td>
      <td>209.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>71.528571</td>
      <td>162.197115</td>
      <td>61.990385</td>
      <td>10.447619</td>
      <td>2.735577</td>
      <td>17.134615</td>
      <td>22.721154</td>
      <td>17.884615</td>
      <td>7.086538</td>
      <td>5.444976</td>
      <td>14.907767</td>
      <td>3.970874</td>
      <td>4.048544</td>
      <td>6.655340</td>
      <td>20.033981</td>
      <td>11.500000</td>
      <td>75.753927</td>
      <td>27.393204</td>
      <td>9.626214</td>
      <td>0.464115</td>
      <td>0.263158</td>
      <td>0.086124</td>
      <td>0.368421</td>
      <td>0.636364</td>
      <td>1.818182</td>
      <td>0.745192</td>
      <td>0.533654</td>
      <td>0.149038</td>
      <td>0.120192</td>
      <td>1.548077</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.603301</td>
      <td>7.883487</td>
      <td>9.816795</td>
      <td>4.537944</td>
      <td>4.714279</td>
      <td>7.302382</td>
      <td>6.397398</td>
      <td>6.062170</td>
      <td>4.918879</td>
      <td>4.428718</td>
      <td>7.280863</td>
      <td>3.675778</td>
      <td>3.676899</td>
      <td>3.453192</td>
      <td>5.911032</td>
      <td>4.671397</td>
      <td>38.405416</td>
      <td>11.153289</td>
      <td>8.782527</td>
      <td>0.796570</td>
      <td>0.666962</td>
      <td>0.328529</td>
      <td>0.761551</td>
      <td>0.833100</td>
      <td>2.617518</td>
      <td>1.006009</td>
      <td>0.878526</td>
      <td>0.483445</td>
      <td>0.380663</td>
      <td>1.960380</td>
    </tr>
    <tr>
      <th>min</th>
      <td>50.000000</td>
      <td>138.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>66.000000</td>
      <td>156.000000</td>
      <td>55.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>11.750000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>16.000000</td>
      <td>8.000000</td>
      <td>51.000000</td>
      <td>23.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>71.000000</td>
      <td>162.000000</td>
      <td>60.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>25.000000</td>
      <td>15.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>15.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>21.000000</td>
      <td>11.500000</td>
      <td>66.000000</td>
      <td>33.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.000000</td>
      <td>169.000000</td>
      <td>68.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>23.000000</td>
      <td>27.000000</td>
      <td>19.250000</td>
      <td>9.250000</td>
      <td>8.000000</td>
      <td>20.000000</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>9.000000</td>
      <td>25.000000</td>
      <td>15.000000</td>
      <td>88.500000</td>
      <td>35.000000</td>
      <td>16.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>92.000000</td>
      <td>182.000000</td>
      <td>92.000000</td>
      <td>20.000000</td>
      <td>27.000000</td>
      <td>29.000000</td>
      <td>30.000000</td>
      <td>54.000000</td>
      <td>26.000000</td>
      <td>21.000000</td>
      <td>31.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>14.000000</td>
      <td>29.000000</td>
      <td>22.000000</td>
      <td>232.000000</td>
      <td>36.000000</td>
      <td>29.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corrmat = df.corr()
plt.figure(figsize=(13, 13))
sns.heatmap(corrmat, vmax=1, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()
```


    
![png](output_25_0.png)
    


I am going to create a Random Forest Model as a baseline to classify between the three classes; Normal, MCI and Dementia. I will be converting the categorical features to numerical features using One Hot encoding and use all features without any feature engineering for the baseline model. I am also setting hyperparameters for the model using heuristics intially. I will use 5-fold cross validation for evaluation. 


```python
#find categorical columns and use one hot encoding to conver them to numerical columns
cat_columns = df_x.select_dtypes(exclude='number').columns
df_x = pd.get_dummies(df_x, columns=cat_columns, drop_first=True)
```


```python
df_x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
      <th>Gender_Male</th>
      <th>History of cerebrovascular disease_Yes</th>
      <th>History of hypertension_Yes</th>
      <th>History of diabetes_Yes</th>
      <th>History of coronary heart disease_Yes</th>
      <th>History of hyperlipidemia_Yes</th>
      <th>History of anemia_Yes</th>
      <th>History of CO poisoning_Yes</th>
      <th>History of general anesthesia during surgery_Yes</th>
      <th>History of abnormal thyroid function_Yes</th>
      <th>History of traumatic brain injury_Yes</th>
      <th>Family history of dementia_Yes</th>
      <th>Smoking history_Yes</th>
      <th>Drinking history_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.0</td>
      <td>165.000000</td>
      <td>90.000000</td>
      <td>9.0</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>17.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>6.00000</td>
      <td>19.000000</td>
      <td>13.0</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>20.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>81.0</td>
      <td>169.000000</td>
      <td>70.000000</td>
      <td>0.0</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>19.000000</td>
      <td>8.0</td>
      <td>75.753927</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.0</td>
      <td>155.000000</td>
      <td>55.000000</td>
      <td>12.0</td>
      <td>1.000000</td>
      <td>22.000000</td>
      <td>28.000000</td>
      <td>14.000000</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>22.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>12.00000</td>
      <td>26.000000</td>
      <td>16.0</td>
      <td>68.000000</td>
      <td>36.000000</td>
      <td>14.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75.0</td>
      <td>169.000000</td>
      <td>75.000000</td>
      <td>9.0</td>
      <td>2.735577</td>
      <td>17.134615</td>
      <td>22.721154</td>
      <td>17.884615</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>14.907767</td>
      <td>3.970874</td>
      <td>4.048544</td>
      <td>6.65534</td>
      <td>20.033981</td>
      <td>11.5</td>
      <td>75.753927</td>
      <td>27.393204</td>
      <td>9.626214</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68.0</td>
      <td>162.197115</td>
      <td>61.990385</td>
      <td>9.0</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>28.000000</td>
      <td>14.000000</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>23.000000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>8.00000</td>
      <td>27.000000</td>
      <td>15.0</td>
      <td>34.000000</td>
      <td>36.000000</td>
      <td>26.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
clf = RandomForestClassifier(max_depth=10, n_estimators=50, random_state=0)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
n_scores = cross_val_score(clf, df_x, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Balanced Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
```

    Balanced Accuracy: 0.871 (0.081)
    

This model yields rather reasonable results, especially as we are looking at Balanced Accuracy which is defined as the average of recall obtained on each class. This is a good measure that takes into account the imbalanced nature of this dataset. However, let us further examine our results by splitting this dataset into a training and test set and further evaluating the performance on the test set.


```python
#use label encoder to convert the categorical labels into numerical ones
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_
```




    array(['Dementia', 'MCI', 'Normal'], dtype=object)




```python
#split the dataset into train and test set and train a random forest classifier on the train set. 
X_train, X_test, y_train, y_test = train_test_split(df_x, y, test_size = 0.2, random_state = 42)
clf2 = RandomForestClassifier(max_depth=10, random_state=0)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
```

    Accuracy: 0.9666666666666667
    


```python
#a set of other metrics are also calculated using confusion matrix. It should be noted that the calculations are the mean of individual binary classification measures.
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
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
```


```python
print("Specificity:", np.mean(TNR))
print("Precision:", np.mean(PPV))
print("Overall Accuracy:", np.mean(ACC))
```

    Specificity: 0.9743589743589745
    Precision: 0.9629629629629629
    Overall Accuracy: 0.9682539682539683
    


```python
#build the confusion matrix using SNS
c_mat = pd.DataFrame(data=np.column_stack((le.inverse_transform(y_test),le.inverse_transform(y_pred))), columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(c_mat['y_Actual'], c_mat['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.savefig('confusion.png', bbox_inches='tight')
plt.show()
```


    
![png](output_35_0.png)
    



```python
#examine the wrongly predicted instances
indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
wrong_predictions = X_test.iloc[indices,:]
wrong_predictions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
      <th>Gender_Male</th>
      <th>History of cerebrovascular disease_Yes</th>
      <th>History of hypertension_Yes</th>
      <th>History of diabetes_Yes</th>
      <th>History of coronary heart disease_Yes</th>
      <th>History of hyperlipidemia_Yes</th>
      <th>History of anemia_Yes</th>
      <th>History of CO poisoning_Yes</th>
      <th>History of general anesthesia during surgery_Yes</th>
      <th>History of abnormal thyroid function_Yes</th>
      <th>History of traumatic brain injury_Yes</th>
      <th>Family history of dementia_Yes</th>
      <th>Smoking history_Yes</th>
      <th>Drinking history_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>69.0</td>
      <td>159.0</td>
      <td>60.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>22.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>87.0</td>
      <td>170.0</td>
      <td>55.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>74.0</td>
      <td>34.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We can see from the confusion matrix that 2 patients with MCI have been predicted as Normal patients. This is reasonable as these patients are closer to Normal than MCI at a glance; for instance, their MMSE is 24 and 27 respectively. If your MMSE is above 24, you are considered to be normal cognition. We could further test this by looking at what are the most important predictors for the model.


```python
#plot the most important features for the model
feature_names = X_train.columns
plt.figure(figsize=(8, 12), dpi=80)
sorted_idx = clf2.feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], clf2.feature_importances_[sorted_idx])
```




    <BarContainer object of 44 artists>




    
![png](output_38_1.png)
    



```python
#get the average values for numerical columns grouped by each class
df.groupby('Label').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
    </tr>
    <tr>
      <th>Label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dementia</th>
      <td>74.775000</td>
      <td>160.675000</td>
      <td>60.150000</td>
      <td>7.475000</td>
      <td>7.875000</td>
      <td>6.615385</td>
      <td>12.375000</td>
      <td>26.575000</td>
      <td>8.550000</td>
      <td>7.850000</td>
      <td>4.894737</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.973684</td>
      <td>12.342105</td>
      <td>5.473684</td>
      <td>118.208333</td>
      <td>13.394737</td>
      <td>0.000000</td>
      <td>0.775000</td>
      <td>0.475000</td>
      <td>0.275000</td>
      <td>1.000000</td>
      <td>1.050000</td>
      <td>3.575000</td>
      <td>0.800000</td>
      <td>0.675000</td>
      <td>0.450000</td>
      <td>0.275000</td>
      <td>2.200000</td>
    </tr>
    <tr>
      <th>MCI</th>
      <td>71.988506</td>
      <td>161.425287</td>
      <td>61.603448</td>
      <td>9.735632</td>
      <td>1.977011</td>
      <td>15.298851</td>
      <td>23.197674</td>
      <td>16.802326</td>
      <td>7.290698</td>
      <td>5.534884</td>
      <td>13.643678</td>
      <td>2.563218</td>
      <td>2.609195</td>
      <td>5.908046</td>
      <td>19.494253</td>
      <td>11.298851</td>
      <td>85.325581</td>
      <td>27.793103</td>
      <td>7.195402</td>
      <td>0.406977</td>
      <td>0.209302</td>
      <td>0.034884</td>
      <td>0.290698</td>
      <td>0.627907</td>
      <td>1.569767</td>
      <td>0.674419</td>
      <td>0.534884</td>
      <td>0.116279</td>
      <td>0.081395</td>
      <td>1.406977</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>69.481928</td>
      <td>163.777778</td>
      <td>63.314815</td>
      <td>12.626506</td>
      <td>1.012346</td>
      <td>24.085366</td>
      <td>27.268293</td>
      <td>14.780488</td>
      <td>6.158537</td>
      <td>4.192771</td>
      <td>20.962963</td>
      <td>7.345679</td>
      <td>7.493827</td>
      <td>9.654321</td>
      <td>24.222222</td>
      <td>14.543210</td>
      <td>53.012346</td>
      <td>33.530864</td>
      <td>16.753086</td>
      <td>0.373494</td>
      <td>0.216867</td>
      <td>0.048193</td>
      <td>0.144578</td>
      <td>0.445783</td>
      <td>1.228916</td>
      <td>0.792683</td>
      <td>0.463415</td>
      <td>0.036585</td>
      <td>0.085366</td>
      <td>1.378049</td>
    </tr>
  </tbody>
</table>
</div>




```python
wrong_predictions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Height</th>
      <th>Weight</th>
      <th>NPI</th>
      <th>MoCAB</th>
      <th>MMSE</th>
      <th>IADL</th>
      <th>HAMA</th>
      <th>HAMD</th>
      <th>C1 HVLT(immediate memory)</th>
      <th>C5 HVLT delayed recall 5min</th>
      <th>C8 HVLT delayed recall 20min</th>
      <th>C4 logical memory WMS</th>
      <th>C6 Boston Naming Test</th>
      <th>C3 articulateness and verbal fluencey-vegetable BNT</th>
      <th>C7-STT_A</th>
      <th>C2 CFT Rey-limitation</th>
      <th>C9 CFT Rey-recall</th>
      <th>HD1 depressive mood</th>
      <th>HD2 guilty</th>
      <th>HD3 suicidal</th>
      <th>HD7 work &amp; interests</th>
      <th>HA6 Total Score of Depressive Mood \n</th>
      <th>Total score of Depression core factors</th>
      <th>HA1 Anxiety</th>
      <th>HA2 Tension</th>
      <th>HA3 Fear</th>
      <th>HA14 Interview perfomance</th>
      <th>Total score of Anxiety factors</th>
      <th>Gender_Male</th>
      <th>History of cerebrovascular disease_Yes</th>
      <th>History of hypertension_Yes</th>
      <th>History of diabetes_Yes</th>
      <th>History of coronary heart disease_Yes</th>
      <th>History of hyperlipidemia_Yes</th>
      <th>History of anemia_Yes</th>
      <th>History of CO poisoning_Yes</th>
      <th>History of general anesthesia during surgery_Yes</th>
      <th>History of abnormal thyroid function_Yes</th>
      <th>History of traumatic brain injury_Yes</th>
      <th>Family history of dementia_Yes</th>
      <th>Smoking history_Yes</th>
      <th>Drinking history_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>69.0</td>
      <td>159.0</td>
      <td>60.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>19.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>22.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>87.0</td>
      <td>170.0</td>
      <td>55.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>74.0</td>
      <td>34.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the feature importance, we can see that MoCAB, MMSE, C1 HVLT(immediate memory), IADL and C8 HVLT delayed recall 20min are the 5 most important feature the model uses to differentiate between the three classes. When we inspect patient 197, their MoCAB is 19 which is in between MCI and Normal; their MMSE is 24 which is closer to the average of Normal patients than those with MCI; their C1 HVLT is 22 which is closer to Normal than MCI; their IADL is 19 which closer to MCI than Normal and their C8 HVLT is 8 which is closer to the Normal than MCI. We can reasonably ascertain this is why the model has predicted patient 197 as Normal rather than a patient with MCI. 

When we inspect patient 162, their MoCAB is 20 which is closer to Normal than MCI; their MMSE is 27 which is closer to the average of Normal patients than those with MCI; their C1 HVLT is 11 which is closer to MCI than Normal; their IADL is 19 which is closer to MCI than Normal and their C8 HVLT is 3 which is closer to MCI than Normal. In this case, top two features are biased towards a Normal patient and the other three features are biased towards MCI. Given that the model considers the first two features as the most important by a significant margin, we can also reasonably ascertain this is why the model has predicted patient 162 as Normal rather than a patient with MCI.  
