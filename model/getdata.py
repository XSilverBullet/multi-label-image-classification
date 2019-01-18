import h5py
import numpy as np

def load():
    # f_train = h5py.File("train_dataset.h5")
    # x_train,y_train = f_train['x'].value, f_train['y'].value

    #print(x_train.shape)
    f_trainval = h5py.File("trainval_dataset.h5")
    x_val, y_val = f_trainval['x'].value, f_trainval['y'].value

    # x_train = np.vstack((x_train, x_val))
    # y_train = np.vstack((y_train, y_val))

    f_test = h5py.File("test_dataset.h5")
    x_test, y_test = f_test['x'].value, f_test['y'].value

    #x_train = np.vstack((x_test, x_val))
    #y_train = np.vstack((y_test, y_val))

    return x_val, y_val, x_test, y_test

# x_train, y_train, x_test, y_test = load()
# print(x_train.shape)

