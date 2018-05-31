import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # for hide logs about something bad in tf

import cv2
import datetime
from termcolor import colored
import numpy as np
from keras.models import load_model

import dataset

from sklearn.preprocessing import LabelBinarizer


def arr2val(arr):
    max_index = 0

    for i, val in enumerate(arr):
        if arr[i] > arr[max_index]:
            max_index = i

    print '+------------------+'
    for i, v in enumerate(arr):
        s = '| %2d) %s  %.4f   |' % (i, chr(ord('A') + i), v)
        if i == max_index:
            print colored(s, 'green')
        else:
            print s

    print '+------------------+'

    return (arr[max_index] * 100)

# Create default image
# with shape 128x128
# and put our image in center
#
def create_default_image(source_image):
    sizze = 128
    default_image = np.empty((sizze, sizze))
    default_image.fill(0)

    if source_image.shape[0] < source_image.shape[1]:
        resize_ratio = sizze / float(source_image.shape[1])
    else:
        resize_ratio = sizze / float(source_image.shape[0])

    image_res_to_def = cv2.resize(source_image, (int(source_image.shape[1] * resize_ratio), int(source_image.shape[0] * resize_ratio)))

    # default_image
    pos_st_y = int((sizze - image_res_to_def.shape[0]) / 2.)
    pos_st_x = int((sizze - image_res_to_def.shape[1]) / 2.)
    default_image[
        pos_st_y:pos_st_y + image_res_to_def.shape[0],
        pos_st_x:pos_st_x + image_res_to_def.shape[1]
    ] = image_res_to_def

    return default_image
#
# ===== ###
#

    
predict_symbol = ''
model = load_model('saved_models/saved_model_large.h5')
model.load_weights('saved_models/saved_weights_large.h5')


image = cv2.imread("dataset_grouped/test/P/P11.jpg")
cv2.imshow('original', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
th_ret, edged = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
edged = cv2.bitwise_not(edged)  # invert

pr_image = create_default_image(edged)
cv2.imshow('pr_image', pr_image)

x_data = pr_image.reshape(1, pr_image.size)

# normalize
x_data /= 255

result = model.predict(x_data, verbose=1)


(X_train, y_train), (X_test, y_test) = dataset.load_data()
encoder = LabelBinarizer()
encoder.fit_transform(y_train)
r_code = encoder.inverse_transform(result)

print "Result: ", chr(r_code)
#for pr in result[0]:
#    print pr*100
for idx, chrcode in enumerate(encoder.classes_):
    print chr(chrcode) + " -> " + str(result[0][idx] * 100)


######################################
cv2.waitKey(0)
cv2.destroyAllWindows()

