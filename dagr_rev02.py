# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:25:14 2019

@author: adsims
"""

from PyQt5.uic import loadUiType

import sys
from PyQt5 import QtWidgets, QtCore, uic
import numpy as np   
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import threading
import dagr_backend

__author__ = "Andrew Sims, Michael Hudson"
__copyright__ = "None"
__license__ = "None"
__version__ = "0.2"
__maintainer__ = "Andrew Sims, Michael Hudson"
__email__ = "andrew.sims.d@gmail.com"
__status__ = "Prototype"
	
Ui_MainWindow, QMainWindow = loadUiType('DAGR.ui')

class Gui(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Gui, self).__init__()
        self.setupUi(self)
        self.dataset_fig_dict = {}
        self.algorithim_fig_dict = {}
        self.learning_curve_fig_dict = {}
        
        self.pb_begin_analysis = self.findChild(QtWidgets.QPushButton, 'pb_begin_analysis')
        self.pb_begin_analysis.clicked.connect(self.begin_analysis)
        
        self.cb_feature_box_and_whisker = self.findChild(QtWidgets.QCheckBox, 'cb_feature_box_and_whisker')
        self.cb_correlation_matrix = self.findChild(QtWidgets.QCheckBox, 'cb_correlation_matrix')
        self.cb_feature_histogram = self.findChild(QtWidgets.QCheckBox, 'cb_feature_histogram')
        self.cb_raw_features = self.findChild(QtWidgets.QCheckBox, 'cb_raw_features')
        self.cb_scatter_matrix = self.findChild(QtWidgets.QCheckBox, 'cb_scatter_matrix')
        
        # self.html_viewer = self.findChild(QtWebEngineWidgets.QWebEngineView, 'webEngineView')
        
        self.dataset_prog = self.findChild(QtWidgets.QProgressBar, 'dataset_prog')
        self.algorithim_prog = self.findChild(QtWidgets.QProgressBar, 'algorithim_prog')
        
        self.cb_adaboost = self.findChild(QtWidgets.QCheckBox, 'cb_adaboost')
        self.cb_dtc = self.findChild(QtWidgets.QCheckBox, 'cb_dtc')
        self.cb_gaussian_process = self.findChild(QtWidgets.QCheckBox, 'cb_gaussian_process')
        self.cb_linear_svm = self.findChild(QtWidgets.QCheckBox, 'cb_linear_svm')
        self.cb_naive_bayes = self.findChild(QtWidgets.QCheckBox, 'cb_naive_bayes')
        self.cb_nearest_neighbors = self.findChild(QtWidgets.QCheckBox, 'cb_nearest_neighbors')
        self.cb_neural_network = self.findChild(QtWidgets.QCheckBox, 'cb_neural_network')
        self.cb_qda = self.findChild(QtWidgets.QCheckBox, 'cb_qda')
        self.cb_random_forest = self.findChild(QtWidgets.QCheckBox, 'cb_random_forest')
        self.cb_rbf_svm = self.findChild(QtWidgets.QCheckBox, 'cb_rbf_svm')
        
        self.dataset_mpl_figs.itemClicked.connect(self.dataset_changefig)
        self.algorithim_mpl_figs.itemClicked.connect(self.algorithim_changefig)
        self.learning_curve_mpl_figs.itemClicked.connect(self.learning_curve_changefig)
        
        datset_base_fig = Figure()
        self.dataset_addmpl(datset_base_fig)
        algorithim_base_fig = Figure()
        self.algorithim_addmpl(algorithim_base_fig)
        learning_curve_base_fig = Figure()
        self.learning_curve_addmpl(learning_curve_base_fig)
    
    def begin_analysis(self,):
        # with open(r'C:/Users/andre/Documents/Data Analysis/bokeh_test_file.html', 'r') as fh:
        #     file = fh.read()
        # self.html_viewer.setHtml(file)
        
        df, feature_names = dagr_backend.gen_dataset()
        dataset_figs = {}
        if self.cb_feature_box_and_whisker.isChecked():
            dataset_figs['Feature Box Plot'] = dagr_backend.plot_feature_box(df, feature_names)
        if self.cb_correlation_matrix.isChecked():
            dataset_figs['Correlation Matrix'] = dagr_backend.plot_corr_mat(df, feature_names)
        if self.cb_feature_histogram.isChecked():
            dataset_figs['Histograms'] = dagr_backend.plot_histograms(df, feature_names)
        if self.cb_raw_features.isChecked():
            dataset_figs['Raw Features'] = dagr_backend.plot_raw_features(df, feature_names)
        if self.cb_scatter_matrix.isChecked():
            dataset_figs['Scatter Matrix'] = dagr_backend.plot_scatter_matrix(df, feature_names)
        
        algorithims_to_use = []
        if self.cb_adaboost.isChecked():
            algorithims_to_use.append('adaboost')
        if self.cb_dtc.isChecked():
            algorithims_to_use.append('dtc') 
        if self.cb_gaussian_process.isChecked():
            algorithims_to_use.append('gaussian_process')
        if self.cb_linear_svm.isChecked():
            algorithims_to_use.append('linear_svm')
        if self.cb_naive_bayes.isChecked():
            algorithims_to_use.append('naive_bayes')
        if self.cb_nearest_neighbors.isChecked():
            algorithims_to_use.append('nearest_neighbors')
        if self.cb_neural_network.isChecked():
            algorithims_to_use.append('neural_network')
        if self.cb_qda.isChecked():
            algorithims_to_use.append('qda')
        if self.cb_random_forest.isChecked():
            algorithims_to_use.append('random_forest')
        if self.cb_rbf_svm.isChecked():
            algorithims_to_use.append('rbf_svm')
        
        models = dagr_backend.build_models(df, feature_names, algorithims_to_use)    
        
        algorithim_fig_list = []
        learning_curve_fig_list = []
        algorithim_fig_list.append(('Algorithim Accuracy', dagr_backend.plot_algorithim_accuracy(df, feature_names, models)))
        for model in models:
            model_name = model.steps[-1][0]
            # x = threading.Thread(target=self.append_figure(), args=(algorithim_fig_list, model_name, dagr_backend.plot_algorithim_class_space(), (df, feature_names, model)))
            # x.start()
            algorithim_fig_list.append((model_name, dagr_backend.plot_algorithim_class_space(df, feature_names, model)))
            learning_curve_fig_list.append(
                (
                    model_name,
                    dagr_backend.plot_learning_curve_(df, feature_names, model)
                )
            )
            
        for i, (name, figure) in enumerate(dataset_figs.items()):
            self.dataset_prog.setValue(((i+1)/len(dataset_figs))*100)
            gui.dataset_addfig(name, figure)
        for i, (name, figure) in enumerate(algorithim_fig_list):
            self.algorithim_prog.setValue(((i+1)/len(algorithim_fig_list))*100)
            gui.algorithim_addfig(name, figure)
        for name, figure in learning_curve_fig_list:
            gui.learning_curve_addfig(name, figure)
            
    def append_figure(lst, model_name, func, func_args):
        lst.append((model_name, func(func_args)))
    def dataset_changefig(self, item):
        text = item.text()
        self.dataset_rmmpl()
        self.dataset_addmpl(self.dataset_fig_dict[text])
 
    def dataset_addfig(self, name, fig):
        self.dataset_fig_dict[name] = fig
        self.dataset_mpl_figs.addItem(name)
        
    def dataset_addmpl(self, fig):
        self.dataset_canvas = FigureCanvas(fig)
        self.dataset_mplvl.addWidget(self.dataset_canvas)
        self.dataset_canvas.draw()
        self.toolbar = NavigationToolbar(self.dataset_canvas, 
                self.dataset_mpl_window, coordinates=True)
        self.dataset_mplvl.addWidget(self.toolbar)
        
    def dataset_rmmpl(self,):
        self.dataset_mplvl.removeWidget(self.dataset_canvas)
        self.dataset_canvas.close()
        self.dataset_mplvl.removeWidget(self.toolbar)
        self.toolbar.close() 
        
    def algorithim_changefig(self, item):
        text = item.text()
        self.algorithim_rmmpl()
        self.algorithim_addmpl(self.algorithim_fig_dict[text])
 
    def algorithim_addfig(self, name, fig):
        self.algorithim_fig_dict[name] = fig
        self.algorithim_mpl_figs.addItem(name)
        
    def algorithim_addmpl(self, fig):
        self.algorithim_canvas = FigureCanvas(fig)
        self.algorithim_mplvl.addWidget(self.algorithim_canvas)
        self.algorithim_canvas.draw()
        self.toolbar = NavigationToolbar(self.algorithim_canvas, 
                self.algorithim_mpl_window, coordinates=True)
        self.algorithim_mplvl.addWidget(self.toolbar)
        
    def algorithim_rmmpl(self,):
        self.algorithim_mplvl.removeWidget(self.algorithim_canvas)
        self.algorithim_canvas.close()
        self.algorithim_mplvl.removeWidget(self.toolbar)
        self.toolbar.close()
   
    def learning_curve_changefig(self, item):
        text = item.text()
        self.learning_curve_rmmpl()
        self.learning_curve_addmpl(self.learning_curve_fig_dict[text])
 
    def learning_curve_addfig(self, name, fig):
        self.learning_curve_fig_dict[name] = fig
        self.learning_curve_mpl_figs.addItem(name)
        
    def learning_curve_addmpl(self, fig):
        self.learning_curve_canvas = FigureCanvas(fig)
        self.learning_curve_mplvl.addWidget(self.learning_curve_canvas)
        self.learning_curve_canvas.draw()
        self.toolbar = NavigationToolbar(self.learning_curve_canvas, 
                self.learning_curve_mpl_window, coordinates=True)
        self.learning_curve_mplvl.addWidget(self.toolbar)
        
    def learning_curve_rmmpl(self,):
        self.learning_curve_mplvl.removeWidget(self.learning_curve_canvas)
        self.learning_curve_canvas.close()
        self.learning_curve_mplvl.removeWidget(self.toolbar)
        self.toolbar.close()
        
    def closeEvent(self, ce):
        #prevents blocking after each run
        QtWidgets.QApplication.quit()
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.processEvents()
    gui = Gui()
    gui.show()
    sys.exit(app.exec_())