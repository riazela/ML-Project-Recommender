import numpy as np


def load_data():
    try:
        a = np.load("jester_npfile.npy")
    except FileNotFoundError:
        print("Binary file not found. Try to load from the text file")
        a = np.loadtxt("jester_ratings.dat")
        a[:, 0:2] = np.round(a[:, 0:2])
        np.save("jester_npfile", a)
        print("Text File loaded and saved as jester_npfile in binary format.")

    X = a[:, 0:2].astype(int)

    Y = a[:, 2]
    return X, Y


def construct_train_matrix(train_x, train_y, test_x, test_y):
    number_of_users = 0
    number_of_items = 0
    for i in np.concatenate((train_x, test_x), 0):
        if (i[0] > number_of_users):
            number_of_users = i[0]
        if (i[1] > number_of_items):
            number_of_items = i[1]

    mat = np.zeros([number_of_users, number_of_items])
    print(train_x.shape)
    print(train_y.shape)
    for i in range(0, np.size(train_x, 0)):
        mat[train_x[i, 0] - 1, train_x[i, 1] - 1] = train_y[i]
    return mat
