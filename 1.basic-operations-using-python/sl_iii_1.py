
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

"""Load the dataset using pandas library"""

df = pd.read_csv("autodata.csv")

"""Check the contents of dataset using **df.head()** and **df.tail()** functions"""

df.head(5)

df.tail(5)

df.info()

df.describe()


df.isnull()

df.isnull().sum()

df.notnull()

df.notnull().sum()

# calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)

"""Calculate the mean value for the 'horsepower' column:"""

avg_hp = df["horsepower"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_hp)

"""Replace "NaN" by mean value:"""

df["horsepower"].replace(np.nan, avg_hp, inplace = True)

"""Calculate the mean value for 'peak-rpm' column:"""

avg_rpm = df["peak-rpm"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_rpm)

"""Replace NaN by mean value:"""

df["peak-rpm"].replace(np.nan, avg_hp, inplace = True)

df['num-of-doors'].value_counts()

df['num-of-doors'].value_counts().idxmax()

#replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "horsepower-binned" column
df.dropna(subset=["horsepower-binned"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

df.isnull().sum()


df['city-L/100km'] = 235/df["city-mpg"]
df.head()

df['highway-L/100km'] = 235/df["highway-mpg"]
df.head()


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

df['height'] = df['height']/df['height'].max()
df[["length","width","height"]].head()

df.columns

df['aspiration'].value_counts()

dummy_variable_1 = pd.get_dummies(df["aspiration"])
dummy_variable_1.head()

"""We now have the value 0 to represent "turbo" and 1 to represent "std" in the column "aspiration". We will now insert this column back into our original dataset."""

df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("aspiration", axis = 1, inplace=True)

df.head()

df["horsepower"]=df["horsepower"].astype(float, copy=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

df["horsepower-binned"].value_counts()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

"""### Peak-rpm"""

df["peak-rpm"]=df["peak-rpm"].astype(float, copy=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["peak-rpm"])

plt.pyplot.xlabel("peak-rpm")
plt.pyplot.ylabel("count")
plt.pyplot.title("Peak-rpm bins")

bins = np.linspace(min(df["peak-rpm"]), max(df["peak-rpm"]), 4)
bins

group_names1 = ['Low', 'Medium', 'High']

df['peakrpm-binned'] = pd.cut(df['peak-rpm'], bins, labels=group_names, include_lowest=True )
df[['peak-rpm','peakrpm-binned']].head(20)

df["peakrpm-binned"].value_counts()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["peakrpm-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Peak-rpm")
plt.pyplot.ylabel("count")
plt.pyplot.title("peak-rpm bins")

"""### Wheel-base"""

df["wheel-base"]=df["wheel-base"].astype(float, copy=True)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["wheel-base"])

plt.pyplot.xlabel("wheel-base")
plt.pyplot.ylabel("count")
plt.pyplot.title("Wheel-base bins")

bins = np.linspace(min(df["wheel-base"]), max(df["wheel-base"]), 4)
bins

group_names = ['Low', 'Medium', 'High']

df['wheelbase-binned'] = pd.cut(df['wheel-base'], bins, labels=group_names, include_lowest=True )
df[['wheel-base','wheelbase-binned']].head(20)

df["wheelbase-binned"].value_counts()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["wheelbase-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Wheelbase")
plt.pyplot.ylabel("count")
plt.pyplot.title("Wheelbase bins")
