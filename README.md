# machine_learning_codes
Code for the machine learning programmes I make
It contains all the code for the machine learning models I make for an online overview

The follwing code can be in tensorflow or keras and sometimes in pytorch
Useful comments are found in the code itself


------------------------------------------------------------------------------------------------------------------------
TO convert the fashion_mnist to mnsit dataset in the fashion_mnsit_analyzer.py
look for the 
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()

and change it with
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

rest of the process remains the same as both these datasets are almost identical and thus not require much changes to be done in the model
