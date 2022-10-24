
from tkinter import Y
from xml.dom.expatbuilder import DOCUMENT_NODE
import  matplotlib.pyplot as plt
from Compare import  DoCompare_DF1_BY_LEAF, DoCompare_DF_BY_LEAF, DoCompare_DNDF_BY_LEAF, DoCompare_DNDT_BY_LEAF, DoCompare_DSVM_BY_LEAF
from DeepSvm import DoProcess_DSVM_Model
#from DeepSvm import Process_DeepSVM
import numpy as np
import lime.lime_tabular
import shap
import leaf
import importlib

from Get_Data import Load_Data, LoadAndMergeData, get_X_Y_data
from DeepForest import Process_DF
from DNDT_DNDF import create_forest_model,run_experiment ,create_tree_model

TYPE_DForest = 1
TYPE_DNDF = 2
TYPE_DNDT = 3
TYPE_DSVM = 4
TYPE_BREAK = -1


DATA_HEADERS = ['Motor_Current','Generator_Current','Engine_Torque','Vehicle_Speed','Generator_Speed','Battery_Current','Engine_Speed']
DATA_CLASSES = ['0 No fault','1 Motor fault','2 Generator fault','3 Battery fault','4 ICE fault']
    

##############################
# Data Load

print("Train Data Loading..")
print("****************")

#split data by 8 : 2 (train : test)
LoadAndMergeData(0.2)
X_train,X_test,Y_train,Y_test = Load_Data(0.2)

print("Data has", X_train.shape[0] ," train data and ",X_test.shape[0] ," test data." )

print("*****************")
print("Data Load Finished")


##############################

while 1:
    print("select the model and estimate LIME and SHAP")
    print("1: Deep Forest")
    print("2: Deep Neural Decision Forest")
    print("3: Deep Neural Decision Tree")
    print("4: Deep SVM")
    print("-1 : quit")
    v = input("please enter a number to select: ")
    v = int(v)
    
    
    print("*" * 30)
    
    # Deep Forest Method
    if v == TYPE_DForest:
        # load train and test data by ratio 8:2
        DF_model = Process_DF(X_train,Y_train,X_test,Y_test)
        print("Deep Forest Training Finished")
        ans = input("Do you continue to estimate Model using SHAP and LEAF?(y/n)")
        if(ans == "y") : 
            DoCompare_DF1_BY_LEAF(DF_model,X_train,Y_train,X_test,Y_test)
        else : break
       
    if v == TYPE_DNDF:
        forest_model = create_forest_model()
        trained_model,test_dataset = run_experiment(forest_model,epochs=3)
        print("*" * 30)
        print("Deep Neural Decision Forest Training Finished")
        ans = input("Do you continue to estimate Model using SHAP and LEAF?(y/n)")
        if(ans == "y") : 
            DoCompare_DNDF_BY_LEAF(trained_model,X_train,Y_train,X_test,Y_test)
        else : break
      
   
    if v == TYPE_DNDT:
        tree_model = create_tree_model()
        trained_model,test_dataset = run_experiment(tree_model,epochs= 3)
        print("*" * 30)
        print("Deep Neural Decision Tree Training Finished")
        ans = input("Do you continue to estimate Model using SHAP and LEAF?(y/n)")
        if(ans == "y") : 
            DoCompare_DNDT_BY_LEAF(trained_model,X_train,Y_train,X_test,Y_test)
        else : break
 
   
    if v == TYPE_DSVM:
        DSVM_model = DoProcess_DSVM_Model(X_train,Y_train,X_test,Y_test,epochs= 20)
        print("*" * 30)
        print("Deep SVM Training Finished")
        ans = input("Do you continue to estimate Model using SHAP and LEAF?(y/n)")
        if(ans == "y") : 
            DoCompare_DSVM_BY_LEAF(DSVM_model,X_train,Y_train,X_test,Y_test)
        else : break
        

    if v == TYPE_BREAK:
        break



