

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

"""**Load the dataset**"""

df= pd.read_csv('Social_Network_Ads.csv')

df.shape

df.head()

"""**Drop the column User ID**"""

df.drop(['User ID'],axis=1,inplace=True)

df.head()

df.Purchased.value_counts()

df.Gender.value_counts()

df.dtypes

"""**Data Preprocessing**"""

df.isnull().sum()

df.describe()

g = sns.catplot(x = "Gender",y = "Purchased",data = df,kind = "bar",height = 4)
g.set_ylabels("Purchased Probability")
plt.show

M2 = pd.crosstab(df.Gender, df.Purchased, normalize='index')
print(M2)
M2.plot.bar(figsize=(6,4),stacked=True)
plt.legend(title='Gender vs Purchased', loc='upper right')
plt.show()

df_without_male = df.drop('Gender', axis=1)
corr = df_without_male.corr()
print(corr.shape)
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')

"""**Get Input Variable into X and Outcome variable into Y**"""

X=df.drop(['Gender','Purchased'],axis=1)
Y= df['Purchased']
X.head()

"""**Split the data into Train set and Test set**"""

from sklearn.model_selection import train_test_split
# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

# Success
print("Training and testing split was successful.")

"""**Build the model**"""

from sklearn.linear_model import LogisticRegression
basemodel= LogisticRegression()
basemodel.fit(X_train,y_train)
print("Training accuracy:", basemodel.score(X_train,y_train)*100)

"""**Make predictions on test data**"""

y_predict= basemodel.predict(X_test)
print("Testing accuracy:", basemodel.score(X_test,y_test)*100)

"""**Normalize the data using Min Max Normalization or any other technique**"""

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

X=df[['Age','EstimatedSalary']]
X_scaled= scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state = 42)
print("Training and testing split was successful.")

model= LogisticRegression()
model.fit(X_train,y_train)
y_predict= model.predict(X_test)
print("Training accuracy:", model.score(X_train,y_train)*100)
print("Testing accuracy:", model.score(X_test,y_test)*100)

"""### Measure the performance of the model

1. Measure the accuracy
2. Measure the perormance using Precision, Recall, Fscore, Support etc.
"""

from sklearn.metrics import accuracy_score
Acc=accuracy_score(y_test,y_predict)
print(Acc)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_predict)
print(cm)

from sklearn.metrics import precision_recall_fscore_support
prf= precision_recall_fscore_support(y_test,y_predict)
print('precision:',prf[0])
print('Recall:',prf[1])
print('fscore:',prf[2])
print('support:',prf[3])

from sklearn.metrics import classification_report
cr= classification_report(y_test,y_predict)
print(cr)

