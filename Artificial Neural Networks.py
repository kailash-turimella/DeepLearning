# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#                       PART-1 - Data preprocessing

# Importing libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


# Importing datasets
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,[-1]].values

# Encoding categorical data into dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder() 
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:,1:]

# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,random_state = 0)

# Feature scaling
#        bringing the age and salary to a specific range
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#                                PART-2 - ANN
#        Importing keras library and packages
#import warnings
#warnings.filterwarnings('ignore')
#import keras
from keras.models import Sequential
from keras.layers import Dense

#        Initialising ANN
classifier = Sequential()

#        Adding one input layer and first hidden layer
classifier.add(Dense(output_dim=6,      # The average of nodes in the input + the output layer. 11+1/2
                     init='uniform',    # Initialise - random value close to zero
                     activation='relu', # Rectifier function
                     input_dim=11))     # Independant variables

#        Adding second hidden layer
classifier.add(Dense(output_dim=6,
                     init='uniform',
                     activation='relu')) # input dim is only necessary in the first hidden layer


# number of nodes in the hidden layer = the average number of nodes in the input and the output layer.
#             (output dim)            = 11+1/2
#      11 - number of independant variables
#      1 - this classification problem has a binary output hence 1
# You can also use parameter tuning to find the best value for output dim

'''
Rectifier function:
   |        
   |        /
   |       / 
   | _____/ ___
'''
#           Adding the output layer
classifier.add(Dense(output_dim=1,          # output dim = number of categories, except when 2
                     init='uniform',        # Initialise - random value close to zero but not zero
                     activation='sigmoid')) # sigmoid gives a probability
# change activation function to softmax if more that two categories present
# no activation function required in regression

#              Compiling the ANN
classifier.compile(optimizer='adam',            # adam - stochastic gradient descent
                   loss='binary_crossentropy',  # if more than two categories present use categorical_crossentropy
                   metrics=['accuracy'])        # metrics expects a list
# optimizer is the algorithm you want to use to find the optimal set of weights in the neural network
# the loss function is a part of stochastic gradient descent required to find the optimal set of weights
# binary_crossentropy is the name of a logrithmic function
# metric - A critarian you choose to eveluate your model


#            Fitting ANN to training set
classifier.fit(X_train,y_train,  # The dependant and the independant variables
               batch_size=10,    # Number of entries that will go through ANN before updating the weights
               nb_epoch=100)     # Rounds through the ANN


#                               PART-3 - Making predictions

#         Predicting test set results
y_pred = classifier.predict(X_test)
#converting probability to true or false
y_pred = (y_pred > 0.5) # if y_pred is greater than 50% it returns true else false

#         Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
