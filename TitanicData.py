from sklearn.datasets import fetch_openml
import seaborn as sns
import matplotlib.pyplot as plt

X,y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
fig = plt.figure(figsize=(10,7))
ax = sns.kdeplot(data=X.age)
plt.show()
print(X)