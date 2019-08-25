import tensorflow as tf
import keras
from keras import backend as K
K.set_image_dim_ordering('th')
(x,y),(x1,y1)=keras.datasets.fashion_mnist.load_data()
x=x/255.0
x1=x1/255.0
x=x.reshape(60000,28,28,1)
x1=x1.reshape(10000,28,28,1)
model=tf.keras.Sequential([
tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation=tf.nn.relu),
tf.keras.layers.MaxPool2D((2,2),strides=2),
tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
tf.keras.layers.MaxPool2D((2,2),strides=2),
tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
tf.keras.layers.MaxPool2D((2,2),strides=2),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128,activation=tf.nn.relu),
tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.estimator.BoostedTreesClassifier(),metrics=['accuracy'])
model.fit(x,y,epochs=10,batch_size=64,validation_data=(x1,y1))
print(x.shape,x1.shape)

