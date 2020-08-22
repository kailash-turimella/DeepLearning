# Convolutional Neural Network

#                                   PART-1 - Building CNN
# Importing libraries
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialising CNN
classifier = Sequential()

#   Step-1 : Convolution (Applying feature detector on the input image to form a feature map)
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
# 32 feature detectors of 3X3 dimension
# the shape of the input image - 64X64 pixels and 3 dimensions(for the three main colours rgb)


#   Step-2 : Pooling  (Reducing the size of the feature map by sliding a sub-table through the feature map)
classifier.add(MaxPooling2D(pool_size=(2,2)))
# dimensions of the sub-table:2X2

# optional - adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#   Step-3 : Flattening  (Converting the pooled feature table into a single vector)
classifier.add(Flatten())


#   Step-4 : Full Conection  (Using the input vector as the input layer of a ANN which will classify the images)
classifier.add(Dense(output_dim=128,activation='relu'))
# Output layer
classifier.add(Dense(output_dim=1,activation='sigmoid')) # Use softmax if more than two categories
# Binary output


# Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# optimizer is the algorithm you want to use to find the optimal number of weights in the neural network
# adam - stochastic gradient descent
# if more than two categories of dependent variables, loss = categorical_crossentropy

#                               PART-2 - Fitting the CNN to the images
# Image augmentation - preprocessing the image to avoid over fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,       #Pixel values are converted to a value between 0 and 1
                                   shear_range = 0.2,      #
                                   zoom_range = 0.2,       #Applying random zooms to the images
                                   horizontal_flip = True) #Images will be horizontally flipped

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',#Path from which images are taken to train
                                                 target_size = (64, 64),#Dimensions of the images (input_shape)
                                                 batch_size = 32,       #32 through CNN before updating the weights 
                                                 class_mode = 'binary') #Dependant variable

test_set = test_datagen.flow_from_directory('dataset/test_set',     #Path from which test images are taken
                                            target_size = (64, 64), #Dimensions of the images(input_shape)
                                            batch_size = 32,        #32 Images through CNN before updating weights
                                            class_mode = 'binary')  #Binary outcome
# Increase Target size in the training and the test set for better accuracy

classifier.fit_generator(training_set,               # Fitting it while also testing it on the training set
                         samples_per_epoch = 8000,   # Number of images in training set
                         nb_epoch = 25,              # Number of epochs to train the CNN
                         validation_data = test_set, # On which we want to evaluate the performance of the CNN
                         nb_val_samples = 2000)      # Number of images in out test set
