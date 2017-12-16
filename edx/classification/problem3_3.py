

from sys import *
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model, neighbors, tree, ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def main():
    inputFile = argv[1]
    outputFile = argv[2]
    # data = np.genfromtxt(inputFile, delimiter=',',skip_header = 1)
    data = np.genfromtxt(inputFile, delimiter=',', skip_header = 1,
                            skip_footer = 0, names = ['x1', 'x2', 'y'])
    X = np.column_stack((data['x1'],data['x2']))
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.4)

    # C = []
    # for c in C:
    # clf = svm.SVC(kernel='linear', C= 100).fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train, cv=5)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    f = open(outputFile,'wt')

    tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('svm_linear,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{'kernel': ['poly'], 'C' : [0.1, 1, 3], 'degree' : [4, 5, 6], 'gamma' : [0.1, 0.5]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('svm_polynomial,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{'kernel': ['rbf'], 'C' : [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('svm_rbf,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{ 'C' : [0.1, 0.5, 1, 5, 10, 50, 100]}]
    clf = GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('logistic,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{ 'n_neighbors' : list(range(1,51,1)), 'leaf_size': list(range(5,61,5))}]
    clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('knn,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{ 'max_depth' : list(range(1,51,1)), 'min_samples_split': list(range(2,11,1))}]
    clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('decision_tree,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    tuned_parameters = [{ 'max_depth' : list(range(1,51,1)), 'min_samples_split': list(range(2,11,1))}]
    clf = GridSearchCV(ensemble.RandomForestClassifier(), tuned_parameters, cv=5)
    clf.fit(X_train, y_train)
    f.write('random_forest,%0.2f,%0.2f' % (clf.best_score_ , clf.score(X_test, y_test)) )

    # plt.scatter(data['x1'],data['x2'], c= data['y'])
    # plt.show()


    
    # f.write(str(alpha) + ",100," + str(beta[0]) + "," + str(beta[1]) + "," + str(beta[2]) + "\n")


if __name__ == "__main__":
	main()



