
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for hide logs about something bad in tf

from keras.models import Model  # basic class for specifying and training a neural network
from keras.layers import Input, Dense  # the two types of neural network layer we will be using
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
from sklearn.preprocessing import LabelBinarizer

from keras.layers import Dropout  # dropout

import dataset

import keras.datasets.mnist as mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#
#
#

batch_size = 128  # in each iteration, we consider 128 training examples at once
num_epochs = 20  # we iterate (twenty) one hundred times over the entire training set
hidden_size = 512  # there will be 512 neurons in both hidden layers

#
#
#

height, width, depth = 128, 128, 1  # images are 128x128 and greyscale
num_classes = 23  # there are 23 classes


(X_train, y_train), (X_test, y_test) = dataset.load_data()

num_train = len(X_train)  #
num_test = len(X_test)  #

#

X_train = X_train.reshape(num_train, height * width)  # Flatten data to 1D
X_test = X_test.reshape(num_test, height * width)  # Flatten data to 1D
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # Normalise data to [0, 1] range
X_test /= 255  # Normalise data to [0, 1] range

#Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
#Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels

encoder = LabelBinarizer()
Y_train = encoder.fit_transform(y_train)
Y_test = encoder.fit_transform(y_test)
#
#
#

inp = Input(shape=(height * width,))  # Our input is a 1D vector of size 784

hidden_1 = Dense(hidden_size, activation='relu')(inp)  # First hidden ReLU layer
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(drop_1)  # Second hidden ReLU layer
drop_2 = Dropout(0.2)(hidden_2)
hidden_3 = Dense(hidden_size, activation='relu')(drop_2)  # Second hidden ReLU layer
out = Dense(num_classes, activation='softmax')(hidden_3)  # Output softmax layer

model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers

#
#
#

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy

#
#
#

model.fit(X_train, Y_train,  # Train the model using the training set...
          batch_size=batch_size, nb_epoch=num_epochs,
          verbose=1, validation_split=0.1)  # ...holding out 10% of the data for validation
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

###
###
###


# serialize model to JSON
model_json = model.to_json()
with open("saved_models/saved_model_large.json", "w") as json_file:
    json_file.write(model_json)

# save
model.save("saved_models/saved_model_large.h5");

# serialize weights to HDF5
model.save_weights("saved_models/saved_weights_large.h5")
print("\n\n")
print("Saved model to disk")


