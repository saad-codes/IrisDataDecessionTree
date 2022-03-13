import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import get_dummies
from sklearn import preprocessing
import seaborn as sns


 
X,y = fetch_openml("house_prices", version=1, as_frame=True, return_X_y=True)
print(X.columns)

# Features Engineering

cols = X.columns
ColInfo = []
for colName in cols:
    ColInfo.append([colName,type(X[colName][0]),100*X[colName].isna().sum()/len(X)])
colmeta = pd.DataFrame(ColInfo,columns= ['columnName',"DataType","PercentageError"])
listOfDropedColumns = list(colmeta[colmeta["PercentageError"]>20]['columnName'])

X = X.drop(listOfDropedColumns,axis = 1)
print(X)


# Total features with missing values

listOfImputedColumns =  list(colmeta[(colmeta["PercentageError"]>0) & (colmeta["PercentageError"] <= 25)]['columnName'])
print(colmeta[colmeta["columnName"].isin(listOfImputedColumns)])
print(X.isna().sum())


# Numerical missing imputations

fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.LotFrontage)
plt.title("LotFrontage")
plt.show()

X.LotFrontage = X.LotFrontage.fillna(X.LotFrontage.median())

fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.LotFrontage)
plt.title("LotFrontage")
plt.show()


fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.MasVnrArea)
plt.title("Imputing missing values using mode")
X.MasVnrArea = X.MasVnrArea.fillna(X.MasVnrArea.median())
_ = sns.kdeplot(X.MasVnrArea)
plt.show()




fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.GarageYrBlt)
plt.title("Imputing missing values using mode")
X.GarageYrBlt = X.GarageYrBlt.fillna(X.GarageYrBlt.median())
_ = sns.kdeplot(X.GarageYrBlt)
plt.show()

# categorical Features imputation

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.MasVnrType)
plt.title("MasVnrArea")
plt.show()

X.MasVnrType = X.MasVnrType.fillna(X.MasVnrType.value_counts().idxmax())


fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.MasVnrType)
plt.title("MasVnrArea")
plt.show()



# BsmtQual

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtQual)
plt.title("BsmtQual")
plt.show()

X.BsmtQual = X.BsmtQual.fillna(X.BsmtQual.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtQual)
plt.title("BsmtQual")
plt.show()

# BsmtCond

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtCond)
plt.title("BsmtCond")
plt.show()

X.BsmtCond = X.BsmtCond.fillna(X.BsmtCond.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtCond)
plt.title("BsmtCond")
plt.show()



# BsmtExposure

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtExposure)
plt.title("BsmtExposure")
plt.show()

X.BsmtExposure = X.BsmtExposure.fillna(X.BsmtExposure.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtExposure)
plt.title("BsmtExposure")
plt.show()


# BsmtFinType1

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtFinType1)
plt.title("BsmtFinType1")
plt.show()

X.BsmtFinType1 = X.BsmtFinType1.fillna(X.BsmtFinType1.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtFinType1)
plt.title("BsmtFinType1")
plt.show()

# BsmtFinType2

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtFinType2)
plt.title("BsmtFinType2")
plt.show()

X.BsmtFinType2 = X.BsmtFinType2.fillna(X.BsmtFinType2.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.BsmtFinType2)
plt.title("BsmtFinType2")
plt.show()

# Electrical

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.Electrical)
plt.title("Electrical")
plt.show()

X.Electrical = X.Electrical.fillna(X.Electrical.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.Electrical)
plt.title("Electrical")
plt.show()

# GarageType

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageType)
plt.title("GarageType")
plt.show()

X.GarageType = X.GarageType.fillna(X.GarageType.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageType)
plt.title("GarageType")
plt.show()



# GarageFinish

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageFinish)
plt.title("GarageFinish")
plt.show()

X.GarageFinish = X.GarageFinish.fillna(X.GarageFinish.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageFinish)
plt.title("GarageFinish")
plt.show()


# GarageQual

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageQual)
plt.title("GarageQual")
plt.show()

X.GarageQual = X.GarageQual.fillna(X.GarageQual.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageQual)
plt.title("GarageQual")
plt.show()


# GarageCond

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageCond)
plt.title("GarageCond")
plt.show()

X.GarageCond = X.GarageCond.fillna(X.GarageCond.value_counts().idxmax())

fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.GarageCond)
plt.title("GarageCond")
plt.show()

# Listfeature Encoding 


listOfCategoricalColumns = []

for i in X.columns:
    if X[i].dtype == "O":
        listOfCategoricalColumns.append(i)

print(listOfCategoricalColumns)

X = get_dummies(X, columns = listOfCategoricalColumns)
print("New Columns are",len(X.columns))



# Data Normalization

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.2)
sel.fit_transform(X)
X = X[X.columns[sel.get_support(indices=True)]] 
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X)

# transform training data
X = norm.transform(X)
print(X)

print(y)
# Data is totally in Numeric form and we can now do normalization

from sklearn import preprocessing
X = preprocessing.normalize(X)
# Import 'train_test_split'
from sklearn.model_selection import train_test_split

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.30, random_state = 29)

# Success

print("Training and testing split was successful.")
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test)
print("accuracy", sum(y_predict)/sum(y_test))
fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(y_test)
_ = sns.kdeplot(y_predict)

plt.title("predicgted")
plt.show()

print(np.sqrt(mean_squared_error(y_test,y_predict)))
fig = plt.figure(figsize=(10,7))
_ = sns.regplot(y_predict,y_test)
plt.show()