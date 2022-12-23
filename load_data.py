import numpy as np

def load_data():
    train = np.load('fashion_train.npy')
    test = np.load('fashion_test.npy')

    X_train, y_train = train[:, :784], train[:, 784]
    X_test, y_test = test[:, :784], test[:, 784]

    return X_train, y_train, X_test, y_test