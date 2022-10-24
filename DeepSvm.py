
import numpy as np
from SVMlayers import Layer
from SVMmodel import Model


import numpy as np

FEATURE_NUMBER = 7

def create_DSVM_Model():
    model = []
    np.random.seed(1) # seed random number generator for reproducible results

    # Set up model layers
    model.append(Layer._input_layer(input_shape = (FEATURE_NUMBER,1)))
    model.append(Layer.dense(input_layer = model[0], num_nodes = 6, rectified = True))
    model.append(Layer.dense(input_layer = model[1], num_nodes = 4, rectified = True))
    model.append(Layer.dense(input_layer = model[2], num_nodes = 2, rectified = True))
    model.append(Layer.dense(input_layer = model[3], num_nodes = 1, rectified = True)) # We need linear support vector machine output layer | Set rectified to false during training

    model[1].learning_rate = 0.001
    model[2].learning_rate = 0.02
    model[3].learning_rate = 0.02
    model[4].learning_rate = 0.02

    return model


def DoProcess_DSVM_Model(X_train,Y_train,X_test,Y_test,epochs = 500):
   
    model = create_DSVM_Model()
    
    print("Start Model Training...")
    # train model using the BSSP learning algorithm
    Model.sgd_bssp_train(model, X_train, Y_train, epochs)

    # rectify the output layer | This line can be commented out
    #model[len(model) - 1].rectified = True
    
    
    #evaluate accuray
    
    len = X_test.shape[0]
    
    cnt = 0
    for i in range(len):
       res = float(Model.predict(model,X_test[i]))
       
       if(round(res) == Y_test[i]) :
            cnt = cnt  + 3.4

    print("Accuracy is : {} %",   cnt * 100 / len)
    
        
    return model
    


    

    


