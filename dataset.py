import numpy as np


def load_data():
    dataset_folder = 'dataset_np'
    x_train = np.load(dataset_folder + '/x_train.npy')
    x_test = np.load(dataset_folder + '/x_test.npy')

    y_train = np.load(dataset_folder + '/y_train.npy')
    y_test = np.load(dataset_folder + '/y_test.npy')

    return (x_train, y_train), (x_test, y_test)