import numpy as np

from sklearn.datasets import load_iris
iris_dataset=load_iris()

print("\n IRIS FEATURES \ TARGET NAMES: \n ", iris_dataset.target_names)
print("\n IRIS DATA :\n",iris_dataset["data"])
print("\n Target :\n",iris_dataset["target"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
 
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)
 
for i in range(len(X_test)):
    x = np.array([X_test[i]])
    prediction = kn.predict(x)
    print("\n Actual : {0} {1}, Predicted :{2}{3}".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][prediction]))

print("\n TEST SCORE[ACCURACY] \n",kn.score(X_test, y_test))