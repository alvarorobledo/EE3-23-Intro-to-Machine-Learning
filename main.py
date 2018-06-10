# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:37:36 2018

@author: Alvaro
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PolynomialFeatures

red_dataset = 'data/red.csv'
red = pd.read_csv(red_dataset, sep=';')

white_dataset = 'data/white.csv'
white = pd.read_csv(white_dataset, sep=';')

data = pd.concat([red, white])

y = data.quality
X = data.drop('quality', axis=1)
#X = PolynomialFeatures(degree=2).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

#from sklearn.decomposition import PCA
#pca = PCA(n_components = 4)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


def baseline():
    from sklearn.dummy import DummyRegressor
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train_scaled, y_train)
    y_pred_train = baseline.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = baseline.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    print (r2_score(y_test, y_pred_test))
    #print (lm.score(X_test_scaled, y_test))
    #plot_conf_mat(y_test, y_pred_round)
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def plot_wine_quality_histogram():
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111) 
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=17)  
    plt.yticks(range(500, 3000, 500), fontsize=17) 
    plt.xlabel("Quality", fontsize=20)  
    plt.ylabel("Count", fontsize=20) 
    plt.hist(y.values, color="#3F5D7D", bins = [3, 4, 5, 6, 7, 8, 9, 10], align='left')
    plt.savefig("histogram.png", bbox_inches='tight')
    
def plot_features_correlation():
    import seaborn as sns
    plt.figure(figsize=(14,14))
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    sns.set(font_scale=1.5)
    corr_mat = data.corr()
    ax = sns.heatmap(data=corr_mat, annot=True, fmt='0.1f', vmin=-1.0, vmax=1.0, center=0.0, square=True, xticklabels=corr_mat.columns, yticklabels=corr_mat.columns, cmap="Blues")
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.figure.tight_layout()
    plt.savefig("features_correlation.png", bbox_inches='tight')
    
def plot_lasso_cv_mse(lr):
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(321)     
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.set_xscale('log')
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)
    ax.set_xlabel('lambda', fontsize=12)
    ax.set_ylabel('mean squared error', fontsize=12)
    plt.plot(lr.alphas_, lr.mse_path_, lw=1)
    plt.savefig('lasso_10cv_mse.png', bbox_inches='tight')

def linear_reg():
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)
    y_pred_train = lm.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = lm.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    print (r2_score(y_test, y_pred_test))
    #print (lm.score(X_test_scaled, y_test))
    #plot_conf_mat(y_test, y_pred_round)
    global metrics_lr
    metrics_lr = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)


def lasso_reg():
    from sklearn.linear_model import LassoCV
    n_alphas = 5000
    alpha_vals = np.logspace(-6, 0, n_alphas)
    lr = LassoCV(alphas=alpha_vals, cv=10, random_state=0)
    lr.fit(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = lr.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    print (lr.alpha_)
    print (lr.score(X_test_scaled, y_test))
    plot_lasso_cv_mse(lr)
    #plot_conf_mat(y_test, pred_round)
    global metrics_lasso
    metrics_lasso = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def elastic_net_reg():
    from sklearn.linear_model import ElasticNetCV
    n_alphas = 300
    l1_ratio = [.1, .3, .5, .7, .9]
    rr = ElasticNetCV(n_alphas=n_alphas, l1_ratio=l1_ratio, cv=10, random_state=0)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = rr.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    print (rr.alpha_, rr.l1_ratio_)
    print (rr.score(X_test_scaled, y_test))
    #plot_conf_mat(y_test, _pred_round)
    global metrics_en
    metrics_en = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def ridge_reg():
    from sklearn.linear_model import RidgeCV
    n_alphas = 100
    alpha_vals = np.logspace(-1, 3, n_alphas)
    rr = RidgeCV(alphas=alpha_vals, cv=10)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = rr.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    print (rr.alpha_)
    print (rr.score(X_test_scaled, y_test))
    #plot_conf_mat(y_test, _pred_round)
    global metrics_ridge
    metrics_ridge = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

#def logistic_reg():
#    from sklearn.linear_model import LogisticRegression
#    loreg = LogisticRegression()
#    loreg.fit(X_train_scaled, y_train)
#    y_pred_train = loreg.predict(X_train_scaled)
#    y_pred_test = loreg.predict(X_test_scaled)
#    print (loreg.score(X_test_scaled, y_test))
#    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def plot_conf_mat(y_test, y_pred_round):
    from sklearn.metrics import confusion_matrix
    confMat = confusion_matrix(y_test, y_pred_round)
    print (confMat)
    
def scores_results(y_train, y_test, y_pred_train, y_pred_test): #accuracy score: with rounding || mse: no rounding
    y_pred_train_round = np.round(y_pred_train)
    y_pred_test_round = np.round(y_pred_test)
    accuracy = [accuracy_score(y_train, y_pred_train_round), accuracy_score(y_test, y_pred_test_round)]
    mse = [mean_squared_error(y_train, y_pred_train), mean_squared_error(y_test, y_pred_test)]
    mse_with_rounding = [mean_squared_error(y_train, y_pred_train_round), mean_squared_error(y_test, y_pred_test_round)]
    results = pd.DataFrame(list(zip(accuracy, mse, mse_with_rounding)), columns = ['accuracy score', 'mse (no rd)', 'mse(w/ rd)'], index = ['train', 'test'])
    return results

#def svm_cl():
#    import time
#    start = time.clock()
#    from sklearn import svm
#    from sklearn.model_selection import GridSearchCV
#    parameters = [{'C': [1, 1000],
#                   'gamma': [0.1, 0.5],
#                   'tol': [0.001]}]
#    clf2 = svm.SVC(kernel = 'rbf')
#    clf = GridSearchCV(estimator = clf2, param_grid = parameters, cv = 5)
#    clf.fit(X_train_scaled, y_train)
#    y_pred_train = clf.predict(X_train_scaled)
#    y_pred_test = clf.predict(X_test_scaled)
#    best_accuracy = clf.best_score_
#    best_parameters = clf.best_params_
#    print (best_accuracy)
#    print (best_parameters)
#    #print_weights(clf2)
#    #clf.
#    print (clf.score(X_test_scaled, y_test))
#    print ('time running: ', time.clock() - start)
#    plot_conf_mat(y_test, y_pred_test)
#    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def svm_reg():
    import time
    start = time.clock()
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [0.1, 1, 10],
                   'epsilon': [0.01, 0.1, 0.5],
                    'gamma': [0.01, 0.1, 0.3, 0.5, 1]}]
    clf2 = svm.SVR(kernel = 'rbf')
    clf = GridSearchCV(clf2, parameters, cv=10)
    clf.fit(X_train_scaled, y_train)
    y_pred_train = clf.predict(X_train_scaled)
    #y_pred_train_round = np.round(y_pred_train)
    y_pred_test = clf.predict(X_test_scaled)
    #y_pred_test_round = np.round(y_pred_test)
    #print_weights(clf)
    best_accuracy = clf.best_score_
    best_parameters = clf.best_params_
    print (best_accuracy)
    print (best_parameters)
    
    print (clf.score(X_test_scaled, y_test))
    print ('time running: ', time.clock() - start)
#    plot_conf_mat(y_test, np.round(y_pred_test))
    global metrics_svm
    metrics_svm = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

def nn_reg():
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    parameters = [{'hidden_layer_sizes': [3, 5, 10, 100],
                   'alpha': [0.0001, 0.01, 1, 10, 100],
                   'activation': ['relu','logistic','tanh', 'identity']}]
    nn = MLPRegressor(solver='lbfgs', random_state=0)
    nn = GridSearchCV(nn, parameters, cv = 10)
    
    nn.fit(X_train_scaled, y_train)
    
    y_pred_train = nn.predict(X_train_scaled)
    y_pred_test = nn.predict(X_test_scaled)
    
#    best_accuracy = nn.best_score_
    best_parameters = nn.best_params_
#    print (best_accuracy)
    print (best_parameters)
    print (nn.score(X_test_scaled, y_test))
    global metrics_nn
    metrics_nn = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)


#def print_weights(clf):
#    weights = pd.DataFrame(list(zip(X_train.columns, clf.dual_coef_)), columns = ['features', 'estimatedCoefficients'])
#    print (weights)


import time
plot_wine_quality_histogram()
plot_features_correlation()
start = time.clock()
print('Linear Regression running...')
print(linear_reg())
print ('time running: ', time.clock() - start)
start = time.clock()
print('Lasso Regression running...')
print(lasso_reg())
print ('time running: ', time.clock() - start)
start = time.clock()
print('Elastic Net Regression running...')
print(elastic_net_reg())
print ('time running: ', time.clock() - start)
start = time.clock()
print('Ridge Regression running...')
print(ridge_reg())
print ('time running: ', time.clock() - start)
start = time.clock()
print('SVM regression running...')
print(svm_reg())
print ('time running: ', time.clock() - start)
start = time.clock()
print('NN regression running...')
print(nn_reg())
print ('time running: ', time.clock() - start)
finalscores = pd.DataFrame(list(zip(metrics_lr, metrics_lasso, metrics_en, metrics_ridge)), columns = ['lr', 'lasso', 'el net', 'ridge'], index = ['acc','mse','r2'])
print('Summary/Comparison of scores')
print(finalscores)