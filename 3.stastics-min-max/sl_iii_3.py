

# Import the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets

iris = datasets.load_iris()
iris

df = pd.DataFrame(iris['data'])
df.head()

df[4] = iris['target']
df.head()

# Adding column names
df.rename(columns = {0:'SepalLengthCm', 1:'SepalWidthCm', 2:'PetalLengthCm', 3:'PetalWidthCm', 4:'Species'}, inplace = True)
df.head()

df.describe()

df.shape

"""**MEAN**"""

df.mean()

"""**MEDIAN**"""

df.median()

"""**MODE**"""

# Calculated only for categorical data
df.Species.mode()

df.groupby(['Species']).count()

"""**STANDARD DEVIATION**"""

df.SepalLengthCm.std()

df.SepalWidthCm.std()

df.PetalLengthCm.std()

df.PetalWidthCm.std()
