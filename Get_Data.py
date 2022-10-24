from cProfile import label
from csv import excel
import sys
from os import  listdir,path
import glob
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

#setting random seed for splitting the data sets
seed = 6

CLASS_NUMBER = 5
DATA_DIRECTORY = "./TrainData"

CSV_HEADERS = ['Motor_Current','Generator_Current','Engine_Torque','Vehicle_Speed','Generator_Speed','Battery_Current','Engine_Speed','result']
RESULT_CLASSES = ['0 No fault','1 Motor fault','2 Generator fault','3 Battery fault','4 ICE fault']


#Load  data from directory By label number
def Load_DataByLabel(label_number):
    dir_fmt = ("label_%s" %label_number);
    loadDir = path.join("./","TrainData",dir_fmt,"./")
    filelist = glob.glob("%s/*.csv" %loadDir) 
    dataset = []
    for file in filelist:
        df = pd.read_csv(file, header=None)
        dataset.append(df)
        
    #concatenate all data into one DataFrame
    all_data_x = pd.concat(dataset,axis = 0,ignore_index=True)
    
    
    #remove empty cells
    all_data_x = all_data_x.dropna()
    
    #remove duplicated rows
    all_data_x.drop_duplicates(inplace = True)    
    
    #create y data by label number
    all_data_y = [[label_number]] * all_data_x.shape[0]
    
    r = pd.DataFrame(all_data_x)
    t = pd.DataFrame(all_data_y)
    tot = pd.merge(r,t)
   
    return all_data_x.to_numpy(),all_data_y
        

#Load and Split the data to train and test data 
def Load_Data(data_split_ratio):
    [X,Y] = Load_DataByLabel(4)
    for i in range(CLASS_NUMBER - 1):
        [x,y] = Load_DataByLabel(i)
        X = np.concatenate((X,x))
        Y = np.concatenate((Y,y))
    
    
    #splitting into train and test sets 
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = data_split_ratio,random_state=seed)
    
  
    
    
    return X_train,X_test,Y_train,Y_test


def Merge2TrainDataByLabel(label_number):
    dir_fmt = ("label_%s" %label_number);
    loadDir = path.join("./","TrainData",dir_fmt,"./")
    
    filelist = glob.glob("%s/*.csv" %loadDir) 
    excel_list = []
    for file in filelist:
        df = pd.read_csv(file, header=None)
        excel_list.append(df)
    

    #create a new dataframe to store the merge excel file
    excel_merged = pd.DataFrame()
    
    for excel_file in excel_list:
        excel_merged = excel_merged.append(excel_file,ignore_index = True)
    
        
    
    excel_merged.to_csv(f"X_{label_number}.csv",index = False, header = False)
    t = [label_number] * excel_merged.shape[0]
    y = pd.DataFrame(t)
    y.to_csv(f"Y_{label_number}.csv",index = False, header = False) 
    #create y data by label number
    all_data_y = [RESULT_CLASSES[label_number]] * excel_merged.shape[0]
   
    
    result = excel_merged.assign(result = all_data_y)
    
    result.columns = CSV_HEADERS

    result.head()
    return result  



#Load and Split the data to train and test data 
def LoadAndMergeData(ratio):
    
    train_path = path.join("./","traindata.csv")
    test_path = path.join("./","testdata.csv")
    total_path = path.join("./","total.csv")
    file_exists = path.exists(train_path) & path.exists(test_path) & path.exists(total_path)
    
    seperate_exist = path.exists("X.csv") & path.exists("Y.csv")
    if(file_exists) :
        print("Train Data is already prepared")
        return

   
    total_list = []
    for i in range(CLASS_NUMBER):
       total_list.append(Merge2TrainDataByLabel(i))
    
    
    total = pd.DataFrame()
    
    for idx_data in total_list:
        total = total.append(idx_data,ignore_index = True)
    
    
    
    print("*" * 30)
    print("Total Data Shape is")
    print(total.shape)
    print("*" * 30)
    total.to_csv(total_path,index = False)
    
    Train_data,Test_data = train_test_split(total,test_size = ratio,random_state=seed)
    
    print("*" * 30)
    print("Train Data Shape is")
    print(Train_data.shape)
    print("*" * 30)
    
    print("*" * 30)
    print("Test Data Shape is")
    print(Test_data.shape)
    print("*" * 30)
    
    Train_data.to_csv(train_path,index = False,header = False)
    Test_data.to_csv(test_path,index = False,header = False)
    
    if(seperate_exist) :
            pass
    else : Create_XY_data()
    
    print("Train and Test Data prepared: train.csv , test.csv")
    
    



def Create_XY_data():
    X = pd.DataFrame()
    Y = pd.DataFrame()
  
    list_x = []
    list_y = []
    for i in range(5):
         path_x  = ("X_%s.csv" %i);
         path_y = ("Y_%s.csv" %i);
         df = pd.read_csv(path_x, header=None)
         list_x.append(df)
         df = pd.read_csv(path_y, header=None)
         list_y.append(df)
    


    #create a new dataframe to store the merge excel file
    X = pd.DataFrame()
    Y = pd.DataFrame()
    for x in list_x:
        X = X.append(x,ignore_index = True)
    for y in list_y:
        Y = Y.append(y,ignore_index = True)
        
    X.to_csv("X.csv",index = False,header = False)
    Y.to_csv("Y.csv",index = False,header = False)    
    
def get_X_Y_data():
    path_x  = ("X.csv");
    path_y = ("Y.csv");
    df_x = pd.read_csv(path_x, header=None)
    df_y = pd.read_csv(path_y,header = None)
    x = pd.DataFrame(df_x)
    y = pd.DataFrame(df_y)
    return x,y