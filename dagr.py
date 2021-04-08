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
        
        self.pb_begin_analysis = self.findChild(QtWidgets.QPushButton, 'pb_begin_analysis')
        self..pb_begin_analysis.itemClicked.connect(self.begin_analysis)
        
        self.cb_feature_box_and_whisker = self.findChild(QtWidgets.QCheckBox, 'cb_feature_box_and_whisker')
        self.cb_correlation_matrix = self.findChild(QtWidgets.QCheckBox, 'cb_correlation_matrix')
        self.ccb_feature_histogram = self.findChild(QtWidgets.QCheckBox, 'cb_feature_histogram')
        self.cb_raw_features = self.findChild(QtWidgets.QCheckBox, 'cb_raw_features')
        self.cb_scatter_matrix = self.findChild(QtWidgets.QCheckBox, 'cb_scatter_matrix')
        
        self.dataset_mpl_figs.itemClicked.connect(self.dataset_changefig)
        self.algorithim_mpl_figs.itemClicked.connect(self.algorithim_changefig)
        
        datset_base_fig = Figure()
        self.dataset_addmpl(datset_base_fig)
        algorithim_base_fig = Figure()
        self.algorithim_addmpl(algorithim_base_fig)
    
    def begin_analysis(self,):
    
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
        
    def closeEvent(self, ce):
        #prevents blocking after each run
        QtWidgets.QApplication.quit()
    
if __name__ == '__main__':
    df, feature_names = dagr_backend.gen_dataset()
    dataset_fig_list = []
    if isChecked(cb_feature_box_and_whisker):
        dataset_fig_list.append(('Raw Features', dagr_backend.plot_raw_features(df, feature_names)))
    dataset_fig_list.append(('Correlation Matrix', dagr_backend.plot_corr_mat(df, feature_names)))
    dataset_fig_list.append(('Feature Box Plot', dagr_backend.plot_feature_box(df, feature_names)))
    dataset_fig_list.append(('Scatter Matrix', dagr_backend.plot_scatter_matrix(df, feature_names)))
    dataset_fig_list.append(('Histograms', dagr_backend.plot_histograms(df, feature_names)))
    
    algorithim_fig_list = []
    algorithim_fig_list.append(('Algorithim Accuracy', dagr_backend.plot_algorithim_accuracy(df, feature_names)))
    
    app = QtWidgets.QApplication(sys.argv)
    gui = Gui()
    for name, figure in dataset_fig_list:
        gui.dataset_addfig(name, figure)
    for name, figure in algorithim_fig_list:
        gui.algorithim_addfig(name, figure)
    gui.show()
    sys.exit(app.exec_())