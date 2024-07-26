import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

def create_dataset(img_folder):
   
    img_data_array=np.empty(shape=[0, 200, 200, 200])
    predict_value=np.empty(shape=[0])
    index = 0
   
    for dir1 in os.listdir(img_folder):
        img_one_array=np.empty(shape=[0, 200, 200])

        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image=cv2.imread( image_path )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255

            # Add an extra dimension to the image
            image = np.expand_dims(image, axis=0)
            img_one_array = np.append(img_one_array, image, axis=0)
        
        img_one_array = np.expand_dims(img_one_array, axis=0)
        predict_value = np.append(predict_value, defined_value[index % 40])
        img_data_array = np.append(img_data_array, img_one_array, axis=0)
        index = index + 1

    return img_data_array, predict_value

def create_model():
    model = Sequential([
        # Input layer
        Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', input_shape=(200, 200, 200, 1)),
        # Output: 200 * 200 * 200 * 8

        # Pooling
        MaxPooling3D(pool_size=(2, 2, 2)),
        # Output: 100 * 100 * 100 * 8

        # Convolution layer
        Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu'),
        # Output: 100 * 100 * 100 * 16

        # Pooling
        MaxPooling3D(pool_size=(4, 4, 4)),
        # Output: 25 * 25 * 25 * 16

        # Convolution layer
        Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'),
        # Output: 25 * 25 * 25 * 32

        # Pooling
        MaxPooling3D(pool_size=(2, 2, 2)),
        # Output: 12 * 12 * 12 * 32

        # Convolution layer
        Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'),
        # Output: 12 * 12 * 12 * 64

        # Convolution layer
        Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu'),
        # Output: 12 * 12 * 12 * 128

        # Flatten
        Flatten(),
        # Output: 22184

        # Dense layers
        Dense(units=128, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    return model

# Metrics Calculation
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mape

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True)

# extract the image array and class name
augmented_data, augmented_labels = create_dataset('Negated_Images_Amplified')

print(len(augmented_data))
print(augmented_labels)

# Initialize lists to store metrics and predictions
mse_scores = []
mape_scores = []
accuracy_scores = []
predicted_values = []

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(augmented_data, augmented_labels, epochs=10, batch_size=32)    

# Iterate over the folds
for train_index, test_index in kf.split(augmented_data):
    # Split the data into training and testing sets for the current fold
    print(train_index)
    print(test_index)

    X_train, X_test = augmented_data[train_index], augmented_data[test_index]
    y_train, y_test = augmented_labels[train_index], augmented_labels[test_index]

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse, mape = calculate_metrics(y_test, y_pred)
    mse_scores.append(mse)
    mape_scores.append(mape)
    
    # Calculate accuracy and store the scores
    accuracy = 100 - mape
    accuracy_scores.append(accuracy)

    predicted_values.extend(y_pred)

# Plotting the accuracy scores
plt.plot(accuracy_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores per Fold')
plt.show()

# Save the weights of the model
model.save_weights('model_weights.h5')

# Step 3: Save the true and predicted values in a CSV file
augmented_labels = augmented_labels / 1000
# predicted_values = predicted_values / 1000

converted_array = np.array([x.item() for x in predicted_values])
converted_array = converted_array / 1000

results = pd.DataFrame({'True Values': augmented_labels, 'Predicted Values': converted_array})
results.to_csv('results.csv', index=False)

mseResults = pd.DataFrame({'mse': mse_scores, 'mape': mape_scores, 'accuracy': accuracy_scores})
mseResults.to_csv('mse_mape.csv', index=False)

# Print the average, minimum, maximum, and standard deviation of the predicted values
print("Average Predicted Value:", np.mean(predicted_values))
print("Minimum Predicted Value:", np.min(predicted_values))
print("Maximum Predicted Value:", np.max(predicted_values))
print("Standard Deviation of Predicted Values:", np.std(predicted_values))

# Step 4: Save the augmented images and update the CSV file
augmented_data = augmented_data.squeeze()  # Remove the singleton dimension
augmented_data = augmented_data.astype(np.uint8)  # Convert the data type to uint8
