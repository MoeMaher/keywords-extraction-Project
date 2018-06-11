import scipy
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

X_train = scipy.sparse.load_npz('X_train.npz')
Y_train = scipy.sparse.load_npz('Y_train.npz')

X_test = scipy.sparse.load_npz('X_test.npz')
Y_test = scipy.sparse.load_npz('Y_test.npz')

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

print(X_train.get_shape())
print(Y_train.get_shape())
print(X_test.get_shape())
print(Y_test.get_shape())

# classifier =BernoulliNB()
#LR = LogisticRegression()
#svC = svm.SVC()
# MLP = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(100,10),random_state=1)

# grid_search on the MLPClassifier to find the best_params_
# alphaa = [0.001+i*0.001 for i in range(0,100)]
# param_grid = [{'hidden_layer_sizes': ([1,2,3,4,5],[1,2,3,4,5]),'activation' : ['identity', 'logistic', 'tanh', 'relu'], 'alpha': alphaa,'solver' : ['lbfgs', 'sgd', 'adam']}]

# grid_search = GridSearchCV(MLP, param_grid, cv=5,scoring='neg_mean_squared_error')
# grid_search.fit(X_train, Y_train.toarray()[0])
# print(grid_search.best_params_)

# classifier.fit(X_train,Y_train.toarray()[0])
# LR.fit(X_train,Y_train.toarray()[0])
# svC.fit(X_train,Y_train.toarray()[0])
# MLP.fit(X_train,Y_train.toarray()[0])

# Y_resultNB = classifier.predict(X_test)
# Y_resultLR = LR.predict(X_test)
# Y_resultSVC = svC.predict(X_test)
# Y_resultMLP = MLP.predict(X_test)


# print("accuracy of Naive Bayes Classifier :",accuracy_score(Y_test.toarray()[0], Y_resultNB))
# print(precision_recall_fscore_support(Y_test.toarray()[0], Y_resultNB, average='macro'))
# print("accuracy of LogisticRegression :",accuracy_score(Y_test.toarray()[0], Y_resultLR))
# print("accuracy of SVM classifier :",accuracy_score(Y_test.toarray()[0], Y_resultSVC))
# print("accuracy of ML classifier :",accuracy_score(Y_test.toarray()[0], Y_resultMLP))



# printing the params of the MLPClassifier

# print(MLP.n_outputs_)
# print(MLP.get_params)

