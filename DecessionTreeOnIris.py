# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
# EDA
# loading data iris data
iris = load_iris()
X = iris.data
y = iris.target
# Distribution
fig = plt.figure(figsize=(25,20))
ax = sns.kdeplot(data = pd.DataFrame(iris.data, columns= iris.feature_names))
plt.title("KDE Plot of features")
plt.xlabel("Values in cm")
plt.show()
# test train split
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
                           cv=7, n_jobs=-1,verbose=1,  scoring = "accuracy")
# Training Data 
grid_search.fit(X_train, y_train)
# training Accuracy
score_df = pd.DataFrame(grid_search.cv_results_)
# getting the estimator that gives best tarining results
dt_best = grid_search.best_estimator_
print("The Accuracy of train Data",dt_best.score(X_train,y_train))
print("Best Classification tree accoring to Gradient Search is \n",dt_best)
# Printing Graphing tree
fig = plt.figure(figsize=(25,20))
_ = plot_tree(dt_best,feature_names=iris.feature_names, class_names=iris.target_names,filled=True)
fig.savefig("decistion_tree.png")

# Saving the best fitted model with name "finalized_model.sav"
filename = 'finalized_model.sav'
pickle.dump(dt_best, open(filename, 'wb'))


# load the model from disk and predicting the test
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)
print("acuuracy for testing data is ", loaded_model.score(X_test,y_test))
from sklearn.metrics import confusion_matrix
print("The confusion Matrix is \n",confusion_matrix(y_test, y_pred))