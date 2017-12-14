'''
    Authors -
    Alesya Trubchik and Divya Agarwal
'''

# Import the libraries required for dong the classification

import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import os
import datetime
import numpy as np
import cv2
import pickle

# For graph visualization
keras.backend.set_image_dim_ordering('tf')
tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)

# Constants required for training the model
batch_size = 16
num_classes = 39
epochs = 12
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'model_trained.h5'

# ------ Collect Data together ------

# change to all_images.pkl for running all images
with open('../preprocessed-vgg/0_stacked_images.pkl', 'rb') as f:
    img_data  = pickle.load(f)
    print 'imagess shape : ', img_data.shape

# change to all_labels.pkl for running all labels
with open('../labels/0_labels.pkl', 'rb') as f:
    labels  = pickle.load(f)
    print 'labels shape : ',labels.shape

# Run the file split_train_test.py to get the pickle files fortraning and testing data

# Load the split training and testing data
train_ind = np.load('train-ind.pkl')
test_ind = np.load('test-ind.pkl')
x_train = img_data[train_ind]
y_train = labels[train_ind]
x_test = img_data[test_ind]
y_test = labels[test_ind]

print 'train samples size: ', x_train.shape[0]
print 'test samples size: ', x_test.shape[0]

# Convert class vectors to binary class matrices.
y_train = y_train - 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = y_test - 1
# Dump the test lables onto the pickle file
pickle.dump(predict, open('y-true-final.pkl', 'wb'))
y_test = keras.utils.to_categorical(y_test, num_classes)


# Convert to ndarray type  float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data Normalization
x_train = x_train / 255.
x_test = x_test / 255.


# ----------------- Build VGG19 model (Architecture) ---------------------

def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model

# Laod the predtrained model weights (link : https://github.com/fchollet/deep-learning-models/releases)
model = VGG_19('vgg19_weights_tf_dim_ordering_tf_kernels.h5')

# Optimizers used
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) # Wining Model
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Test 1
rms  = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0) # Test 2
sgd2 = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True) # Test 3

# Compile the Model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Get the weights for every layers
model_weights_full = model.get_weights()

# ----------- My MODEL -----------------------------

def my_vgg19():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Model
model = my_vgg19()

# Get the weights for every layers
mymodel_weights_full = model.get_weights()

# Optimizers used
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) # Wining Model
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Test 1
rms  = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0) # Test 2
sgd2 = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True) # Test 3

# Compile the Model
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#set weigths for training
weights_list = model_weights_full
weights_list[-1] = mymodel_weights_full[-1]
weights_list[-2] = mymodel_weights_full[-2]
model.set_weights(weights_list)

# -------- Uncomment for second run --------
# Load pre-trained model weights
#model.load_weights('model_weights.h5')

# Start the timer
start_time = datetime.datetime.now()
print "Start Time is : ",start_time

# Training the Model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs = epochs,
          validation_split=0.17,
          shuffle = True,
          callbacks = [tb_callback])

# End the timer
end_time = datetime.datetime.now()
print("End Time is : ",end_time)

total_time = end_time - start_time
print("Total time is :", total_time)

#Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    model.save_weights('model4_weights_0001.h5')
    print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Predict and save predictions
#predict = model.predict(x_test) ---> Propabilites
predict = model.predict_classes(x_test) # class lables

print 'test set shape', x_test.shape
print 'predict shape', predict.shape

pickle.dump(predict, open('y-pred-final.pkl', 'wb'))
    
    
# Visualize the model Architecture
from keras.utils import plot_model
plot_model(model, to_file='model4.png')


