# This project is simply compare 3 algorithm from sklearn library
# on guessing sexuality from training inputs.

from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes

clf_tree = tree.DecisionTreeClassifier()

clf_neighbors = neighbors.KNeighborsClassifier()

clf_naive_bayes = naive_bayes.GaussianNB()

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
#Using decision tree
clf_tree.fit(X,Y)
prediction_tree = clf_tree.predict([[190, 70, 43],[160, 70, 43]])

# Using K_mean nearing clustering
clf_neighbors.fit(X,Y)
prediction_neighbor = clf_neighbors.predict([[190, 70, 43],[160, 70, 43]])

#Using Naive_bayes algorithm
clf_naive_bayes.fit(X,Y)
prediction_naive_bayes = clf_naive_bayes.predict([[190, 70, 43],[160, 70, 43]])


print(prediction_tree)

