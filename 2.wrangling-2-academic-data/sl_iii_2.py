
# Commented out IPython magic to ensure Python compatibility.
# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Read the csv file
data = pd.read_csv("tecdiv.csv")

data

print("The first five rows are as follows: ")
data.head()

print("The last five rows are as follows: ")
data.tail()

# Describtion about the dataset
data.describe()

# Information about the dataset
data.info()

print("The column names of the dataset are as follows: ")
data.columns

data.isnull().sum()

"""Here, we can see that there are no null values. Hence, there is no need of data cleaning or replacing NULL values"""

# Converting the roll numbers from TECOC342 --> 342

# if this dosent work
# for i in data['Roll no '].iteritems():

# try
for i in data['Roll no '].items():
    data.loc[i[0], 'Roll no '] = data.loc[i[0], 'Roll no '][-3:]

data.head()

"""## Outliers"""

sns.boxplot(y=data['First year:   Sem 1'])

sns.boxplot(y=data['First year:   Sem 2'])

sns.boxplot(y=data["Second year:   Sem 1"])

sns.boxplot(y=data["Second year:   Sem 2"])

"""Here, we have visualized the outliers for the results column in the dataset"""
