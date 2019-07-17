# to code the fashion mnist dataset
import tensorflow as tf
import numpy as np
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
#loading the dataset from the default libary in tensorflow 
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
#reshaping the data according to the input layer in the neural network
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',input_shape=(28,28,1),activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D((2,2),strides=2),
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D((2,2),strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    
])
#defining the dense layers and neural network architectute as well as the input shape for the first layer
x_train=x_train/255.0
x_test=x_test/255.0
#normalizing the parameters in the dataset so that it would be easier for the neural network to process
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
#making the labeled data categorically as the fashion mnist dataset is divided into 10 labels of various categories
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])
#compiling the model
model.fit(x_train,y_train,batch_size=64,verbose=1,validation_data=(x_train,y_train),epochs=10)
#optimizing the model with the trainig data and using validation data to test the accuracy of the neural network
print(x_train.shape)
print(y_train.shape)