from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# survival - Survival (0 = No; 1 = Yes)
# class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name - Name
# sex - Sex
# age - Age
# sibsp - Number of Siblings/Spouses Aboard
# parch - Number of Parents/Children Aboard
# ticket - Ticket Number
# fare - Passenger Fare
# cabin - Cabin
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# boat - Lifeboat (if survived)
# body - Body number (if did not survive and body was recovered)

# # loading data 

X,y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Printing Any NAN values
print("total percentage of NaN in Features \n", 100*X.isna().sum()/(len(X)))


# Droping 'name','cabin','boat','home.dest','ticket'  columns 

X = X.drop(['name',
            'cabin',
            'boat',
            'home.dest',
            'body',
            'ticket'], axis = 1)

print("total percentage of NaN in after removing Features with high percentage of missing values \n", 100*X.isna().sum()/(len(X)))
# Feature engineering embarked people

# Making countplot of embarked people before imputation
fig = plt.figure(figsize=(10,7))
_ = sns.countplot(X.embarked)
plt.xlabel("embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)")
plt.ylabel("Number of pessenges embarked")
plt.ylim((0,len(X)))
plt.title("Count plot passenger embarking from port")
plt.show()
# imputing Maximum Value
X.embarked = X.embarked.fillna(X.embarked.value_counts().idxmax())


print("total percentage of NaN in after removing NANs from embarked \n", 100*X.isna().sum()/(len(X)))

# Ploting the Distribution plot for fare 
fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.fare)
plt.title("Density of Fare")
plt.show()
# since It is right skewed so 
X.fare = X.fare.fillna(X.fare.median())
print("total percentage of NaN in after removing NANs from Fare \n", 100*X.isna().sum()/(len(X)))

# Ploting the Distribution plot for age 

fig = plt.figure(figsize=(10,7))
_ = sns.kdeplot(X.age)
plt.title("Density of age")
plt.show()
# since It is right skewed so 
X.age = X.age.fillna(X.age.median())
print("total percentage of NaN in after removing NANs from age \n", 100*X.isna().sum()/(len(X)))

# Tacking the features with categorical features

X = get_dummies(X, columns = ['sex', 'embarked'])
print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 50, test_size = 0.24)

# Setting parameter
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

# making a decession tree clasifier and making a grid search
treeClass = DecisionTreeClassifier(random_state=50)
grid_search = GridSearchCV(estimator=treeClass, 
                           param_grid=params, 
                           cv=15, n_jobs=-1,verbose=1,  scoring = "accuracy")
# Training Data 
grid_search.fit(X_train, y_train)
# training Accuracy
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df)
# getting the estimator that gives best tarining results
dt_best = grid_search.best_estimator_
print("The Accuracy of train Data",dt_best.score(X_train,y_train))
print("Best Classification tree accoring to Gradient Search is \n",dt_best)
# Printing Graphing tree
fig = plt.figure(figsize=(25,20))
_ = plot_tree(dt_best)
plt.show()

# # Saving the best fitted model with name "finalized_model.sav"
# filename = 'finalized_model.sav'
# pickle.dump(dt_best, open(filename, 'wb'))


# # load the model from disk and predicting the test
# loaded_model = pickle.load(open(filename, 'rb'))
# y_pred = loaded_model.predict(X_test)
# print("acuuracy for testing data is ", loaded_model.score(X_test,y_test))
# from sklearn.metrics import confusion_matrix
# print("The confusion Matrix is \n",confusion_matrix(y_test, y_pred))