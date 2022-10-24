#from deepforest import CascadeForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def Create_DF_Model():

    #model = CascadeForestClassifier()
    model = RandomForestClassifier(n_estimators = 100) 
    return model

def Process_DF(X,Y,X_Test,Y_Test):  
    print("Deep Forest Model Created")
    clf = Create_DF_Model()
    clf.fit(X,Y)
    
    # plot the tree structure
    
    #predict from the decision tree
    Y_pred = clf.predict(X_Test)
    #and calculate the accuracy from test database and predict data
    # Model Accuracy, how often is the classifier correct?
    accuracy = accuracy_score(Y_Test,Y_pred)

    print("Deep Forest Classifier accuracy score is : {} %".format(accuracy * 100))
    return clf    
    
    


