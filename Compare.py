






from statistics import mode
from tkinter import Y
import  matplotlib.pyplot as plt
from DNDT_DNDF import create_forest_model
from DeepNeuralDecisionTree import Create_DNDT_Model, Process_DNDT
from Get_Data import get_X_Y_data
from SVMmodel import Model
import numpy as np
import shap
import pandas as pd
import leaf
import importlib
import lime.lime_tabular
import sklearn
import xgboost
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf

DATA_HEADERS = ['Motor_Current','Generator_Current','Engine_Torque','Vehicle_Speed','Generator_Speed','Battery_Current','Engine_Speed']
DATA_CLASSES = ['0 No fault','1 Motor fault','2 Generator fault','3 Battery fault','4 ICE fault']


def DoCompare_DNDT_BY_LEAF(model,X_train,Y_train,X_test,Y_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names = DATA_HEADERS,class_names = DATA_CLASSES ,mode='classification')
    model = Process_DNDT(X_train,Y_train,X_test,Y_test)
    exp = explainer.explain_instance(X_test[44328],model.predict_proba, num_features=5)

    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    p = exp.as_list()
    print(p)
    
    print("Please wait a moment")
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    print("Display shap diagram about the DNDT")
    shap.summary_plot(shap_values, X_test)


   
def DoCompare_DNDF_BY_LEAF(model,X_train,Y_train,X_test,Y_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names = DATA_HEADERS,class_names = DATA_CLASSES ,mode='classification')
    model = Process_DNDT(X_train,Y_train,X_test,Y_test)
    exp = explainer.explain_instance(X_test[1231],model.predict_proba, num_features=5)

    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    p = exp.as_list()
    print(p)
    
    print("Please wait a moment")
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    print("Display shap diagram about the DNDF")
    shap.summary_plot(shap_values, X_train)
   


    


def DoCompare_DSVM_BY_LEAF(model,X_train,Y_train,X_test,Y_test):
    
    print("*" * 30)
    print("#" * 30)
    
    model = sklearn.svm.SVC(kernel='rbf', gamma=0.5, C=0.1)

    #X,Y = get_X_Y_data()

    X_test  =X_test[0:150]
    Y_test  =Y_test[0:150]
    
    X_train = X_train[0:3000]
    Y_train = Y_train[0:3000]
    model.fit(X_train,Y_train)
    
  
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names = DATA_HEADERS,class_names = DATA_CLASSES ,mode='classification')
    model = Process_DNDT(X_train,Y_train,X_test,Y_test)
    exp = explainer.explain_instance(X_test[80],model.predict_proba, num_features=5)

    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    p = exp.as_list()
    print(p)
    # explain all the predictions in the test set
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_train)
    
    importlib.reload(leaf)
    
    X,Y = get_X_Y_data()

    X.columns = DATA_HEADERS

    X.head()
    l = leaf.LEAF(model, X, DATA_HEADERS)

    l.explain_instance(X.iloc[0], num_reps=50)


    print('LIME stability:        ', l.get_lime_stability())
    print('LIME local_concordance:', l.get_lime_local_concordance())
    print('LIME fidelity:         ', l.get_lime_fidelity())
    print('LIME prescriptivity:   ', l.get_lime_prescriptivity())
    print()
    print('SHAP stability:        ', l.get_shap_stability())
    print('SHAP local_concordance:', l.get_shap_local_concordance())
    print('SHAP fidelity:         ', l.get_shap_fidelity())
    print('SHAP prescriptivity:   ', l.get_shap_prescriptivity())
    
    
    
   

def DoCompare_DF1_BY_LEAF(model,X_train,Y_train,X_test,Y_test):
    
    print("*" * 30)
    print("#" * 30)

    X_test  =X_test[0:150]
    Y_test  =Y_test[0:150]
    
    X_train = X_train[0:3000]
    Y_train = Y_train[0:3000]
    model.fit(X_train,Y_train)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names = DATA_HEADERS,class_names = DATA_CLASSES ,mode='classification')
    model = Process_DNDT(X_train,Y_train,X_test,Y_test)
    exp = explainer.explain_instance(X_test[80],model.predict_proba, num_features=5)
    
    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    p = exp.as_list()
    print(p)
    X,Y = get_X_Y_data()

    X_test  =X.iloc[10:100]
    Y_test  =Y.iloc[10:100]
    
    X_train = X.iloc[0:1000]
    Y_train = Y.iloc[0:1000]
    

    

    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_train)
    
    
    importlib.reload(leaf)
    X.columns = DATA_HEADERS

    X.head()
    l = leaf.LEAF(model, X, DATA_HEADERS)

    l.explain_instance(X.iloc[0], num_reps=50)


    print('LIME stability:        ', l.get_lime_stability())
    print('LIME local_concordance:', l.get_lime_local_concordance())
    print('LIME fidelity:         ', l.get_lime_fidelity())
    print('LIME prescriptivity:   ', l.get_lime_prescriptivity())
    print()
    print('SHAP stability:        ', l.get_shap_stability())
    print('SHAP local_concordance:', l.get_shap_local_concordance())
    print('SHAP fidelity:         ', l.get_shap_fidelity())
    print('SHAP prescriptivity:   ', l.get_shap_prescriptivity())
    
    
    
def DoCompare_DF_BY_LEAF(model,X_train,Y_train,X_test,Y_test):
    
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names = DATA_HEADERS,class_names = DATA_CLASSES ,mode='classification')
    exp = explainer.explain_instance(X_test[44328],model.predict_proba, num_features=7)

    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    plt.savefig('LIME_result_DeepForest.png')
    p = exp.as_list()
    print(p)
    
    X,y = shap.datasets.adult()
    X_display,y_display = shap.datasets.adult(display=True)
    print("Please wait a moment")
    X_test = pd.DataFrame(X_test)
    shap.initjs()
    explainer = shap.KernelExplainer(model.predict, X_test.iloc[:50,:])
    shap_values = explainer.shap_values(X_test.iloc[299,:], nsamples=500)
    shap.plots.bar(shap_values)
    exp = shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[299,:])
    exp.as_pyplot_figure()

    plt.tight_layout()
    plt.show()
    p = exp.as_list()
    