import os
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.tree
import sklearn.neural_network
import dill as pickle
import xgboost

def model_train_and_save(model_name, dataset, dataset_name):

    np.random.seed(1)
    if model_name == 'lr': # Logistic Regression
        classifier = sklearn.linear_model.LogisticRegression(random_state=1, n_jobs=5, penalty='l1', solver='saga', tol=0.01)
    elif model_name == 'dt': # Decision Tree
        classifier = sklearn.tree.DecisionTreeClassifier(random_state=1, max_depth=6)
    elif model_name == 'nn': # Neural Network Classifier
        classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50))
    elif model_name == 'xgb': # XGBoost Classifier
        classifier = xgboost.XGBClassifier(n_estimators=400, nthread=10, seed=1)
    else:
        raise NotImplementedError('This model has not been implemented!')

    classifier.fit(dataset.train, dataset.labels_train)
    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, classifier.predict(dataset.train)))
    print('Validation', sklearn.metrics.accuracy_score(dataset.labels_validation, classifier.predict(dataset.validation)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, classifier.predict(dataset.test)))
    pickle.dump(classifier.predict(dataset.validation), open(os.path.join('data', dataset_name, 'dev', f'{model_name}_labels') + '.pkl', 'wb'))
    pickle.dump(classifier.predict(dataset.test), open(os.path.join('data', dataset_name, 'test', f'{model_name}_labels') + '.pkl', 'wb'))
    pickle.dump(classifier, open(os.path.join('data', dataset_name, 'models', f'{model_name}_model.pkl'), 'wb'))
    return classifier