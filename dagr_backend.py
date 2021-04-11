# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:49:13 2019

@author: andre
"""

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from itertools import combinations 
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from matplotlib import cm as color_maps
from numpy import set_printoptions
import timeit
import pathlib
import os
import matplotlib
matplotlib.use('Agg')

__author__ = "Andrew Sims, Michael Hudson"
__copyright__ = "None"
__license__ = "None"
__version__ = "0.2"
__maintainer__ = "Andrew Sims, Michael Hudson"
__email__ = "andrew.sims.d@gmail.com"
__status__ = "Prototype"


def gen_dataset():
    num_samples = 500
    base_time = datetime.datetime(2019, 1, 1)
    time_series = np.array([base_time + (datetime.timedelta(seconds=i))/10 for i in range(num_samples)])
    altitude = np.random.normal(scale = 1.0, size = num_samples)*2000 + 10000
    for i, element in enumerate(altitude):
        altitude[i] = element * (-np.cos(2*np.pi*i/num_samples) + 1)
    temperature = (altitude.max() - altitude)/1000 + np.random.poisson(5, size = num_samples)
    humidity = temperature + np.random.exponential(scale = 1, size = num_samples)*10
    pressure = abs(((temperature / (32 + altitude)) + np.random.pareto(5, size = num_samples))**3)
    pitch = np.random.normal(scale = 1.0, size = num_samples)
    roll = np.random.normal(scale = 1.0, size = num_samples)
    yaw = np.random.normal(scale = 1.0, size = num_samples)
    latitude = np.random.normal(scale = 1.0, size = num_samples) + 30
    longitude = np.random.normal(scale = 1.0, size = num_samples) - 180
    #########################################
    #             CLASSIFY DATA             #
    #########################################
    fail_conditions = 'True Fail Conditions:\n'
    pressure_mod = pressure - np.percentile(pressure, 75)
    fail_conditions += f'Pressure > {np.percentile(pressure, 75)}\n'
    temperature_mod = temperature - np.percentile(temperature, 70)
    fail_conditions += f'Temperature > {np.percentile(temperature, 90)}'
    classification = []
    for _pressure, _temperature in zip(pressure_mod, temperature_mod):
        if (_pressure > 0) | (_temperature > 0):
            classification.append(0)
        else:
            classification.append(1)
#    classification = pd.Series(classification, dtype="category")        
    class_names = ['Fail', 'Pass']
    data = {
            'time':time_series,
            'temperature':temperature,
            'humidity':humidity,
            'altitude':altitude,
            'pressure':pressure,
            # 'pitch' : pitch,
            # 'roll' : roll,
            # 'yaw' : yaw,
            # 'latitude' : latitude,
            # 'longitude' : longitude,
            'classification': classification
            } 
    feature_names = list(data.keys())
    feature_names.remove('time')
    feature_names.remove('classification')
    df = pd.DataFrame(data)
    return (df, feature_names)

def plot_raw_features(df, feature_names):
    fig = plt.figure()
    ncols = np.ceil(len(feature_names)/2)
    for i, feature in enumerate(feature_names):
        plt.subplot(2, ncols, i + 1)
        plt.plot(df['time'], df[feature])
        plt.xticks(rotation=45)
        plt.xlabel('time')
        plt.ylabel(feature)
    plt.suptitle('Raw Features')
    plt.close()
    return fig
    

def plot_corr_mat(df, feature_names):
    corr = df[feature_names].corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.cm.RdBu
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap = cmap)
    plt.title('Feature Correlation')
    labels = feature_names.copy()
    labels.insert(0, '')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.close()
    return fig  

def plot_feature_box(df, feature_names):
    ax = df[feature_names].plot(kind='box')
    fig = ax.get_figure()
    plt.suptitle('Box & Whisker Plot')
    plt.close()
    return fig

def plot_scatter_matrix(df, feature_names):
    plt.style.use('default')
    scatter_matrix(df[feature_names], diagonal='kde')
    plt.suptitle('Scatter Matrix')
    return plt.gcf()

def plot_histograms(df, feature_names):
    df[feature_names].hist(bins = 20)
    plt.suptitle('Histograms')
    return plt.gcf()    

def build_models(df, feature_names, algorithims_to_use):
    algorithims = []
    
    if 'adaboost' in algorithims_to_use:
        algorithims.append(("AdaBoost", AdaBoostClassifier())) 
    if 'dtc' in algorithims_to_use:
        algorithims.append(("Decision Tree", DecisionTreeClassifier(max_depth=5)))
    if 'gaussian_process' in algorithims_to_use:
        algorithims.append(("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))))
    if 'linear_svm' in algorithims_to_use:
        algorithims.append(("Linear SVM", SVC(kernel="linear", C=0.025)))
    if 'naive_bayes' in algorithims_to_use:
        algorithims.append(("Naive Bayes", GaussianNB()))
    if 'nearest_neighbors' in algorithims_to_use:
        algorithims.append(("Nearest Neighbors", KNeighborsClassifier(3)))
    if 'neural_network' in algorithims_to_use:
        algorithims.append(("Neural Net", MLPClassifier(alpha=1, max_iter=1000)))
    if 'qda' in algorithims_to_use:
        algorithims.append(("QDA", QuadraticDiscriminantAnalysis()))
    if 'random_forest' in algorithims_to_use:
        algorithims.append(("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
    if 'rbf_svm' in algorithims_to_use:
        algorithims.append(("RBF SVM", SVC(gamma=2, C=1)))

    # create feature union
#    features = []
#    features.append(('pca', PCA(n_components='mle')))
#    features.append(('select_best', SelectKBest(k=2)))
#    feature_union = FeatureUnion(features)

    estimators = []
#    estimators.append(('feature_union', feature_union))
#    estimators.append(('pca', PCA(n_components = 'mle')))
    estimators.append(('standardize', StandardScaler()))
    
    models = []
    for algorithim in algorithims:
        models.append(Pipeline(estimators + [algorithim]))
 
    return models   
        
def plot_algorithim_accuracy(df, feature_names, models):
    # Split-out validation dataset
    X = df[feature_names]
    Y = df['classification']
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        
    scoring = 'accuracy'
    # evaluate each model in turn
    results = []
    names = []  
    for model in models:
        name = model.steps[-1][0]
        kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)  

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_xticklabels(names, rotation=40, ha='right') 
    return fig


def plot_learning_curve_(df, feature_names, model):
    name = model.steps[-1][0]
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = f'Learning Curve\n{name}'
    fig = plot_learning_curve(model, title, df[feature_names], df['classification'],
                                ylim=(0.7, 1.01), cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)) 
    return fig


def plot_algorithim_class_space(df, feature_names, clf):
    coutnour_step_size = 200
    comb = list(combinations(feature_names, 2))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    name = clf.steps[-1][0]
    individual_figure = plt.figure(figsize=(16, 10))
    individual_figure.suptitle(f'Classification Space of {name}')
    for pairidx, pair in enumerate(comb):
        # We only take the two corresponding features
        X = df[list(pair)]
        y = df['classification']
                
        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
    
        #set plot step so there are 200 steps per axis
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        x_step = (x_max - x_min)/coutnour_step_size
        y_step = (y_max - y_min)/coutnour_step_size
        xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                             np.arange(y_min, y_max, y_step))
    
        # just plot the dataset first
        individual_ax = individual_figure.add_subplot(2, np.ceil(len(comb)/2), pairidx + 1)
        individual_ax.set_xlabel(pair[0])
        individual_ax.set_ylabel(pair[1])
        individual_ax.set_xlim(xx.min(), xx.max())
        individual_ax.set_ylim(yy.min(), yy.max())

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        individual_ax.contourf(xx, yy, Z, cmap=cm, alpha = 1)
        # Plot the training points
        individual_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        individual_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        individual_ax.text(0.97, 0.03, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', bbox=dict(facecolor = 'white', alpha=0.5), transform=individual_ax.transAxes)   
    # #decision tree plot
    # clf = DecisionTreeClassifier().fit(df[feature_names], df['classification'])
    # plt.style.use('default')
    # plt.figure(figsize = [16, 16])
    # ###TODO this plot is broken. pdf? GraphViz?
    # plot_tree(clf, filled=True, class_names = class_names, feature_names = feature_names)
    # plt.annotate(fail_conditions, xy=(0, 1))
    return individual_figure

     
    
def main():
    #########################################
    #                 INPUTS                #
    #########################################
    main_start_time = timeit.default_timer()
    save_results = True
    do_combined_plot = True # very expensive with num_samples > 1000
    do_learning_curve = False
    img_dpi = 600
    plt.style.use('default')
    num_samples = 500
    coutnour_step_size = 200
    set_printoptions(precision=2)
    
    save_path_root = pathlib.Path.cwd() / 'Artifacts' / 'Pipelined'
    save_path_root.mkdir(parents=True, exist_ok=True)
    report_file_path = save_path_root / 'Performance_Report.txt'
    with open(report_file_path, 'w', encoding='utf-8') as fh:
        fh.write('MACHINE LEARNING ALGORITHIM PERFORMANCE REPORT\n' \
                 f'REPORT DATE: {datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}\n' \
                 f'File: {__file__}\n' \
                 f'Author: {__author__}\n' \
                 f'Copyright: {__copyright__}\n' \
                 f'License: {__license__}\n' \
                 f'Version: {__version__}\n' \
                 f'Maintainer: {__maintainer__}\n' \
                 f'E-Mail: {__email__}\n' \
                 f'Status: {__status__}\n\n')
        
    if save_results:
        print(f'Results will be saved in the following directory\n{save_path_root}')
#    plt.style.use('ggplot')
    #########################################
    #            GENERATE DATASET           #
    #########################################
    base_time = datetime.datetime(2019, 1, 1)
    time_series = np.array([base_time + (datetime.timedelta(seconds=i))/10 for i in range(num_samples)])
    altitude = np.random.normal(scale = 1.0, size = num_samples)*2000 + 10000
    for i, element in enumerate(altitude):
        altitude[i] = element * (-np.cos(2*np.pi*i/num_samples) + 1)
    temperature = (altitude.max() - altitude)/1000 + np.random.poisson(5, size = num_samples)
    humidity = temperature + np.random.exponential(scale = 1, size = num_samples)*10
    pressure = abs(((temperature / (32 + altitude)) + np.random.pareto(5, size = num_samples))**3)
    pitch = np.random.normal(scale = 1.0, size = num_samples)
    roll = np.random.normal(scale = 1.0, size = num_samples)
    yaw = np.random.normal(scale = 1.0, size = num_samples)
    latitude = np.random.normal(scale = 1.0, size = num_samples) + 30
    longitude = np.random.normal(scale = 1.0, size = num_samples) - 180
    #########################################
    #             CLASSIFY DATA             #
    #########################################
    fail_conditions = 'True Fail Conditions:\n'
    pressure_mod = pressure - np.percentile(pressure, 75)
    fail_conditions += f'Pressure > {np.percentile(pressure, 75)}\n'
    temperature_mod = temperature - np.percentile(temperature, 70)
    fail_conditions += f'Temperature > {np.percentile(temperature, 90)}'
    classification = []
    for _pressure, _temperature in zip(pressure_mod, temperature_mod):
        if (_pressure > 0) | (_temperature > 0):
            classification.append(0)
        else:
            classification.append(1)
#    classification = pd.Series(classification, dtype="category")        
    class_names = ['Fail', 'Pass']
    data = {
            'time':time_series,
            'temperature':temperature,
            'humidity':humidity,
            'altitude':altitude,
            'pressure':pressure,
            'pitch' : pitch,
            'roll' : roll,
            'yaw' : yaw,
            'latitude' : latitude,
            'longitude' : longitude,
            'classification': classification
            } 
    feature_names = list(data.keys())
    feature_names.remove('time')
    feature_names.remove('classification')
    df = pd.DataFrame(data)
    #########################################
    #            NORMALIZE DATA             #
    #########################################
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    for feature in feature_names:
#        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1,1))
#    
#    cov = EmpiricalCovariance().fit(df[feature_names])
#    with open(report_file_path, 'a', encoding='utf-8') as fh:
#        fh.write(f'Features:\n{df.describe()}\n' \
#                   f'Empirical Covariance:\n{cov.covariance_}\n\n')
    #####################################
    #          DATASET PLOTTING         #
    #####################################
    #raw data
    fig = plt.figure()
    ncols = np.ceil(len(feature_names)/2)
    for i, feature in enumerate(feature_names):
        plt.subplot(2, ncols, i + 1)
        plt.plot(df['time'], df[feature])
        plt.xticks(rotation=45)
        plt.xlabel('time')
        plt.ylabel(feature)
    plt.suptitle('Raw Features')
    if save_results:
        save_path = save_path_root / 'Dataset' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Raw_Features.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    #correlation matrix
    corr = df[feature_names].corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.cm.RdBu
    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap = cmap)
    plt.title('Feature Correlation')
    labels = feature_names.copy()
    labels.insert(0, '')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
    plt.show()
    if save_results:
        save_path = save_path_root / 'Dataset' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Correlation_Matrix.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    #box and whisker
    df[feature_names].plot(kind='box')
    plt.suptitle('Box & Whisker Plot')
    plt.show()
    if save_results:
        save_path = save_path_root / 'Dataset' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Box_and_Whisker_Plot.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    # scatter plot matrix
    plt.style.use('default')
    scatter_matrix(df[feature_names], diagonal='kde')
    plt.suptitle('Scatter Matrix')
    plt.show()
    plt.style.use('ggplot')
    if save_results:
        save_path = save_path_root / 'Dataset' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Scatter_Matrix.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    # histograms
    df[feature_names].hist(bins = 20)
    plt.suptitle('Histograms')
    plt.show()
    if save_results:
        save_path = save_path_root / 'Dataset' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Histograms.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    ###############################
    #          ALGORITHIMS        #
    ###############################
    # Split-out validation dataset
    X = df[feature_names]
    Y = df['classification']
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    
    # Spot Check Algorithms
    algorithims = []
    algorithims.append(("Nearest Neighbors", KNeighborsClassifier(3)))
    algorithims.append(("Linear SVM", SVC(kernel="linear", C=0.025)))
    algorithims.append(("RBF SVM", SVC(gamma=2, C=1)))
    algorithims.append(("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))))
    algorithims.append(("Decision Tree", DecisionTreeClassifier(max_depth=5)))
    algorithims.append(("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
    algorithims.append(("Neural Net", MLPClassifier(alpha=1, max_iter=1000)))
    algorithims.append(("AdaBoost", AdaBoostClassifier()))
    algorithims.append(("Naive Bayes", GaussianNB()))
    algorithims.append(("QDA", QuadraticDiscriminantAnalysis()))
    
    # create feature union
#    features = []
#    features.append(('pca', PCA(n_components='mle')))
#    features.append(('select_best', SelectKBest(k=2)))
#    feature_union = FeatureUnion(features)

    estimators = []
#    estimators.append(('feature_union', feature_union))
#    estimators.append(('pca', PCA(n_components = 'mle')))
    estimators.append(('standardize', StandardScaler()))
    
    models = []
    for algorithim in algorithims:
        models.append(Pipeline(estimators + [algorithim]))
        #        # create feature union
#        features = []
#        features.append(('pca', PCA(n_components=3)))
#        features.append(('select_best', SelectKBest(k=6)))
#        feature_union = FeatureUnion(features)
#        # create pipeline
#        estimators = []
#        estimators.append(('feature_union', feature_union))
#        estimators.append(('logistic', LogisticRegression(solver='liblinear')))
#        model = Pipeline(estimators)
        # Test options and evaluation metric
        
    scoring = 'accuracy'
    # evaluate each model in turn
    results = []
    names = []
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Training Models')   
    for model in models:
        name = model.steps[-1][0]
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        val_start = timeit.default_timer()
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        val_elapsed = timeit.default_timer() - val_start
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} {name} validation duration: {val_elapsed:.2f}')
        results.append(cv_results)
        names.append(name)
        train_start = timeit.default_timer() 
        model.fit(X_train, Y_train)
        train_elapsed = timeit.default_timer() - train_start
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} {name} training duration: {train_elapsed:.2f}')
        predictions = model.predict(X_validation)  
        if do_learning_curve:
            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            title = f'Learning Curve\n{name}'
            plot_learning_curve(model, title, df[feature_names], df['classification'],
                                ylim=(0.7, 1.01), cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
            if save_results:
                save_path = save_path_root / 'Algorithims' / 'Learning Curves'
                save_path.mkdir(parents=True, exist_ok=True)
                file_path = save_path / f'{name}_Learning_Curve.png'
                plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
                plt.close()
#        print(f'{name} Accuracy:\n\tmean: {cv_results.mean()}\n\tstd: {cv_results.std()}')        
#        print(accuracy_score(Y_validation, predictions))
#        print(confusion_matrix(Y_validation, predictions))
#        print(classification_report(Y_validation, predictions))
        with open(report_file_path, 'a', encoding='utf-8') as fh:
            fh.write(f'------------Algorithim: {name}------------\n' \
                     f'Model Parameters:\n{model}\n' \
                     f'Accuracy Score:\n{accuracy_score(Y_validation, predictions)}\n' \
                     f'Confisuion Matrix:\n{confusion_matrix(Y_validation, predictions)}\n' \
                     f'Classification Report:\n{classification_report(Y_validation, predictions)}\n' \
                     f'Validation Time:\n{val_elapsed:.2f}\n' \
                     f'Training Time:\n{train_elapsed:.2f}\n')
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    ax.set_xticklabels(names, rotation=40, ha='right')
    plt.show()
    if save_results:
        save_path = save_path_root / 'Algorithims' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / 'Algorithim_Box_and_Whisker.png'
        plt.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
        plt.close()
    
    comb = list(combinations(feature_names, 2))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ######################################
    #            COMBINED PLOT           #
    ######################################
    if do_combined_plot:
        i = 1
        combined_figure = plt.figure(figsize=(27, 9))
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Plotting Combined Classification Space Plot')
        start_time = timeit.default_timer()
        for pairidx, pair in enumerate(comb):
            # We only take the two corresponding features
            X = df[list(pair)]
            y = df['classification']
                    
            # preprocess dataset, split into training and test part
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.4, random_state=42)
        
            #set plot step so there are 200 steps per axis
#            x_step = (X[0].max() - X[0].min())/coutnour_step_size
#            y_step = (X[1].max() - X[1].min())/coutnour_step_size
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            x_step = (x_max - x_min)/coutnour_step_size
            y_step = (y_max - y_min)/coutnour_step_size
            xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                                 np.arange(y_min, y_max, y_step))
        
            # just plot the dataset first
            combined_ax = combined_figure.add_subplot(len(comb), len(models) + 1, i)
            if pairidx == 0:
                combined_ax.set_title("Input data")
            combined_ax.set_xlabel(pair[0])
            combined_ax.set_ylabel(pair[1])
            # Plot the training points
            combined_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            combined_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                       edgecolors='k')
            combined_ax.set_xlim(xx.min(), xx.max())
            combined_ax.set_ylim(yy.min(), yy.max())
            combined_ax.set_xticks(())
            combined_ax.set_yticks(())
            i += 1
        
            # iterate over classifiers
            for clf in models:
                name = clf.steps[-1][0]
                combined_ax = plt.subplot(len(comb), len(models) + 1, i)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
        
                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                combined_ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        
                # Plot the training points
                combined_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                combined_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                           edgecolors='k', alpha=0.6)
        
                combined_ax.set_xlim(xx.min(), xx.max())
                combined_ax.set_ylim(yy.min(), yy.max())
                combined_ax.set_xticks(())
                combined_ax.set_yticks(())
                if pairidx == 0:
                    combined_ax.set_title(name)
                combined_ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right', bbox=dict(facecolor = 'white', alpha=0.5))
                i += 1
        
        combined_figure.suptitle("Machine Learning Algorithim Classification Space By Algorithim")
        if save_results:
            save_path = save_path_root / 'Algorithims' 
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / 'Combined_Algorithim_Class_Space.png'
            combined_figure.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
            plt.close()
        elapsed = timeit.default_timer() - start_time
        with open(report_file_path, 'a', encoding='utf-8') as fh:
                fh.write(f'Combined Classification Space Plot Run Time: {elapsed:.2f} Seconds\n')

    ######################################
    #          INDIVIDUAL PLOTS          #
    ######################################
    # iterate over classifiers
    for clf in models:
        name = clf.steps[-1][0]
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Plotting {name} Individual Classification Space Plot')
        start_time = timeit.default_timer()
        individual_figure = plt.figure(figsize=(16, 10))
        individual_figure.suptitle(f'Classification Space of {name}')
        for pairidx, pair in enumerate(comb):
            # We only take the two corresponding features
            X = df[list(pair)]
            y = df['classification']
                    
            # preprocess dataset, split into training and test part
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.4, random_state=42)
        
            #set plot step so there are 200 steps per axis
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            x_step = (x_max - x_min)/coutnour_step_size
            y_step = (y_max - y_min)/coutnour_step_size
            xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                                 np.arange(y_min, y_max, y_step))
        
            # just plot the dataset first
            individual_ax = individual_figure.add_subplot(2, np.ceil(len(comb)/2), pairidx + 1)
            individual_ax.set_xlabel(pair[0])
            individual_ax.set_ylabel(pair[1])
            individual_ax.set_xlim(xx.min(), xx.max())
            individual_ax.set_ylim(yy.min(), yy.max())

            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
    
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            individual_ax.contourf(xx, yy, Z, cmap=cm, alpha = 1)
            # Plot the training points
            individual_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            individual_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                       edgecolors='k')
            individual_ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right', bbox=dict(facecolor = 'white', alpha=0.5))
        if save_results:
            save_path = save_path_root / 'Algorithims' / 'Individual Model Performance'
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f'{name}_Class_Space.png'
            individual_figure.savefig(file_path, dpi = img_dpi, bbox_inches = 'tight')
            plt.close()
        elapsed = timeit.default_timer() - start_time
        with open(report_file_path, 'a', encoding='utf-8') as fh:
            fh.write(f'{name} Classification Space Plot Run Time: {elapsed:.2f} Seconds\n')   
    #decision tree plot
    clf = DecisionTreeClassifier().fit(df[feature_names], df['classification'])
    plt.style.use('default')
    plt.figure(figsize = [16, 16])
    ###TODO this plot is broken. pdf? GraphViz?
    plot_tree(clf, filled=True, class_names = class_names, feature_names = feature_names)
    plt.annotate(fail_conditions, xy=(0, 1))
    plt.style.use('ggplot')
    plt.show()
    if save_results:
        save_path = save_path_root / 'Algorithims' 
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f'Decision_Tree.png'
        plt.savefig(file_path)
        plt.close()
    main_elapsed = timeit.default_timer() - main_start_time
    print(f'Total Run Time: {main_elapsed:.2f} Seconds')
    with open(report_file_path, 'a', encoding='utf-8') as fh:
        fh.write(f'Total Run Time: {main_elapsed:.2f} Seconds\n') 


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.style.use('default')
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return fig    

   
if __name__ == "__main__":
    main()        
    print('Done')
    
