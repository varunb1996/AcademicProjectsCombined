from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


#Create Decision Tree Classifier Model and return
def Create_DNDT_Model():

    DNDT_model = tree.DecisionTreeClassifier(criterion='entropy',
                splitter = "best",
                min_samples_split = 3,
                min_samples_leaf = 3,
                max_features = "sqrt",
                class_weight = "balanced")
    
    return DNDT_model

#Train and return Trained Model
def Process_DNDT(X,Y,X_Test,Y_Test):  
      
    clf = Create_DNDT_Model()
    clf = clf.fit(X,Y)
    
    return clf    
    
    


