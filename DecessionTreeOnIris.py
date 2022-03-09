# Importing Data from Sklearn data sets
# Data set name is iris 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 50, test_size = 0.24)

# Data representation

# df = pd.DataFrame(features,columns=iris.feature_names)
# df["target"] = label
# print(df.sample(10))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(
    ccp_alpha=0.01,
    max_depth=7,
    min_samples_split=11, 
    criterion='entropy',
    min_samples_leaf=3)
model = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))
fig = plt.figure(figsize=(25,20))
_ = plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)
fig.savefig("decistion_tree.png")
