## importing libraries
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


## functions
def Logistic_Regression(train, target, test=None, cv=5, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, solver='liblinear', max_iter=100, seed=123, metric='auc'):
    """Performs k-fold logistic regression.
    Returns train and test dataframes with predictions
    """

    # evaluation metric
    def score(a, b, metric=metric):
        if metric == 'auc':
            return roc_auc_score(np.array(a), np.array(b))

    # preparing data
    X_test = pd.DataFrame.copy(test)

    # model
    lr = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, solver=solver, max_iter=max_iter, random_state=seed)

    # stratified k-fold
    kfolds = StratifiedKFold(np.array(target), n_folds=cv, random_state=seed)

    # cross-validation
    print('Logistic Regression')
    print('%d-fold Cross-Validation' % (int(cv)))

    k = 1

    for build_index, val_index in kfolds:
        X_build, X_val, y_build, y_val = train.ix[build_index], train.ix[val_index], target.ix[build_index], target.ix[val_index]
        lr.fit(np.array(X_build), np.array(y_build))
        
        X_val['pred_lr'] = lr.predict_proba(X_val)[:,1]
        X_test['pred_lr'] = lr.predict_proba(np.array(test))[:,1]

        if k == 1:
            train_lr = X_val[:]
            test_lr = X_test[:]

        if k > 1:
            train_lr = pd.concat([train_lr, X_val])
            test_lr['pred_lr'] = ((k-1) * test_lr['pred_lr'] + X_test['pred_lr']) / k

        print('CV Fold-%d %s: %f' % (int(k), str(metric), score(y_val, X_val['pred_lr'])))

        k += 1

    print('\nLogistic Regression %d-fold %s: %f' % (int(cv), str(metric), score(target, train_lr['pred_lr'])))

    return train_lr, test_lr



