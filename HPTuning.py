import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt    
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH = 200, 200, 200
defined_value=[
    0.573,
    0.562,
    0.5795,
    0.5276,
    0.579,
    0.6115,
    0.5767,
    0.56,
    0.568,
    0.5505,
    0.5628,
    0.5241,
    0.5894,
    0.5738,
    0.5892,
    0.5439,
    0.5489,
    0.5692,
    0.5353,
    0.54,
    0.5572,
    0.5592,
    0.5725,
    0.5376,
    0.5438,
    0.5517,
    0.52,
    0.5575,
    0.5544,
    0.546,
    0.6317,
    0.6108,
    0.555,
    0.603,
    0.6311,
    0.6313,
    0.6106,
    0.6388,
    0.643,
    0.652,
]

class DataGenerator(Sequence):
    def __init__(self, dir_list, defined_values, img_folder='Negated_Images_Amplified', batch_size=8, shuffle=True):
        self.img_folder = img_folder
        self.defined_values = defined_values
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dir_list = dir_list
        self.indexes = np.arange(len(self.dir_list))

    def __len__(self):
        return len(self.dir_list) // self.batch_size

    def __getitem__(self, index):
        print('Length of dir_list:', len(self.dir_list))
        print('Batch size:', self.batch_size)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_dirs = [self.dir_list[k] for k in indexes]
        X, y = self.__data_generation(batch_dirs)
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_dirs):
        X = np.empty((self.batch_size, 200, 200, 200, 1))
        y = np.empty((self.batch_size, 1))
        for i, dir_name in enumerate(batch_dirs):
            dir_path = os.path.join(self.img_folder, dir_name)
            file_list = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
            image_array = np.empty((200, 200, 200))
            for j, file_name in enumerate(file_list):
                image = cv2.imread(os.path.join(dir_path, file_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (200, 200), interpolation = cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                image_array[:, :, j] = image
            X[i,] = np.expand_dims(image_array, axis=3)
            y[i] = self.defined_values[int(dir_name[2:]) % 40]
        return X, y

    
    
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=True,
    verbose=1, 
    mode='auto'
)
    
# using gradient checkpoint to prevent memory failure.
class CheckpointedConv3DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, kernel_regularizer=None, **kwargs):
        super(CheckpointedConv3DLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.conv3d = Conv3D(self.filters, self.kernel_size, kernel_regularizer=self.kernel_regularizer)
        super(CheckpointedConv3DLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.recompute_grad(self.conv3d)(inputs)


class CheckpointedDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, kernel_regularizer=None, **kwargs):
        super(CheckpointedDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.dense = Dense(self.units, activation=self.activation, kernel_regularizer=self.kernel_regularizer)
        super(CheckpointedDenseLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.recompute_grad(self.dense)(inputs)



    

def create_model():
    model = Sequential([
        # Input layer
        Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', input_shape=(200, 200, 200, 1), kernel_regularizer=l2(0.01)),
        # Output: 200 * 200 * 200 * 8

        # Pooling
        MaxPooling3D(pool_size=(2, 2, 2)),
        # Output: 100 * 100 * 100 * 8

        # Convolution layer
        Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        # Output: 100 * 100 * 100 * 16

        # Pooling
        MaxPooling3D(pool_size=(4, 4, 4)),
        # Output: 25 * 25 * 25 * 16

        # Convolution layer
        Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        # Output: 25 * 25 * 25 * 32

        # Pooling
        MaxPooling3D(pool_size=(2, 2, 2)),
        # Output: 12 * 12 * 12 * 32

        # Convolution layer
        Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        # Output: 12 * 12 * 12 * 64

        # Convolution layer
        Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        # Output: 12 * 12 * 12 * 128

        # Global Max Pooling
        GlobalMaxPooling3D(),
        # Output: 128

        # Flatten
        Flatten(),
        # Output: 128

        # Dense layers
        Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    
    return model


# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Metrics Calculation
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mape

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True)

# Initialize the data generator
data_gen = DataGenerator(dir_list=[f for f in os.listdir('Negated_Images_Amplified') if os.path.isdir(os.path.join('Negated_Images_Amplified', f))], defined_values=defined_value, batch_size=8)

# Split the directories into training, validation, and test sets
train_val_dirs, test_dirs = train_test_split(data_gen.dir_list, test_size=0.2, random_state=42)


# Split the data into training and testing sets for the current fold
test_gen = DataGenerator(dir_list=test_dirs, defined_values=defined_value, batch_size=8)


#define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    LearningRateScheduler(scheduler),
    checkpoint
]





for train_index, val_index in kf.split(train_val_dirs):
    train_dirs = [train_val_dirs[i] for i in train_index]
    val_dirs = [train_val_dirs[i] for i in val_index]
    train_gen = DataGenerator(dir_list=train_dirs, defined_values=defined_value, batch_size=8)
    val_gen = DataGenerator(dir_list=val_dirs, defined_values=defined_value, batch_size=8)

    

# Define initial coarse search space
param_grid = {
    'learning_rate': [1e-2, 1e-3, 1e-4],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'dense_units': [64, 128, 256],
    'l2_rate': [0.0, 0.01, 0.1],
    'optimizer': ['adam', 'rmsprop', 'sgd']
}
grid = ParameterGrid(param_grid)

best_model = None
best_val_loss = np.inf
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Perform the Coarse Search
for params in grid:
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    
    model = create_model(
        learning_rate=params['learning_rate'], 
        dropout_rate=params['dropout_rate'], 
        dense_units=params['dense_units'], 
        l2_rate=params['l2_rate'], 
        optimizer=optimizer
    )
    model.fit(train_gen, validation_data=val_gen, epochs=50, verbose=1, callbacks=callbacks)
    val_loss = model.history.history['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_params = params

# Define fine search space around best parameters from coarse search
param_grid = {
    'learning_rate': [best_params['learning_rate'] * i for i in [0.5, 0.75, 1, 1.25, 1.5]],
    'dropout_rate': [best_params['dropout_rate'] * i for i in [0.5, 0.75, 1, 1.25, 1.5]],
    'dense_units': [best_params['dense_units'] + i for i in [-32, 0, 32]],
    'l2_rate': [best_params['l2_rate'] * i for i in [0.5, 0.75, 1, 1.25, 1.5]],
    'optimizer': [best_params['optimizer']]
}
grid = ParameterGrid(param_grid)

# Perform the Fine Search
for params in grid:
    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=params['learning_rate'])
    
    model = create_model(
        learning_rate=params['learning_rate'], 
        dropout_rate=params['dropout_rate'], 
        dense_units=params['dense_units'], 
        l2_rate=params['l2_rate'], 
        optimizer=optimizer
    )
    model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks)
    val_loss = model.history.history['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_params = params

# Write the results to a text file
with open('results.txt', 'w') as f:
    f.write("Best Hyperparameters: " + str(best_params) + "\n")
    f.write("Validation Loss: " + str(best_val_loss) + "\n")
    f.write("Training Loss: " + str(best_model.history.history['loss'][-1]) + "\n")

