import os
import pandas as pd
import numpy as np
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt


def performPCA(X, normalize = True, use_SVD=False):
    '''
    :param X: input matrix (countries x features)
    :param standardize: if true then divide by standard diviation)
    :return: first 2 principal components (PC1, PC2)
    '''
    m = X.shape[0]
    ### below part is taken from demo code
    if normalize:
        std_X = np.std(X, axis=0)
        X_norm = X @ np.diag(np.ones(std_X.shape[0]) / std_X)
        X_norm = X_norm.T
    else:
        X_norm = X.T

    # start PCA
    # center the data (also taken from democode
    mu = np.mean(X_norm, axis=1)
    X_norm = X_norm - mu[:, None]

    K = 2
    if use_SVD:
        W, S, Vt = ll.svds(X_norm, k=K)
    else:
        # compute the covariance matrix
        C = np.dot(X_norm, X_norm.T) / m
        # compute the eigen decomposition and get 1st 2 eigenvalues and vectors
        S, W = ll.eigs(C, k=K)
        S = S.real
        W = W.real


    # compute PC1, PC2
    PC1 = np.dot(W[:, 0].T, X_norm) / np.sqrt(S[0])
    PC2 = np.dot(W[:, 1].T, X_norm) / np.sqrt(S[1])

    return PC1, PC2


def plot_q2a():
    '''
       main function for Q2 a
    '''
    file_path = os.getcwd() + '\\data\\food-consumption.csv'
    df = pd.read_csv(file_path)
    X = df.iloc[:, 1:].to_numpy()
    PC1, PC2 = performPCA(X, normalize=False, use_SVD=True)

    # plot
    fig, ax = plt.subplots()
    countries = df.iloc[:, 0]

    ax.scatter(PC1, PC2)
    for i, label in enumerate(countries):
        ax.annotate(label, (PC1[i], PC2[i]))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Countries')
    plt.savefig('PCA_q2a.png')
    #plt.show()

def plot_q2b():
    '''
    main function for Q2 b
    '''
    file_path = os.getcwd() + '\\data\\food-consumption.csv'
    df = pd.read_csv(file_path)
    X = df.iloc[:, 1:].to_numpy()
    X = X.T
    PC1, PC2 = performPCA(X, normalize=True, use_SVD=True)

    # plot
    fig, ax = plt.subplots()
    foods = df.columns[1:]

    ax.scatter(PC1, PC2)
    for i, label in enumerate(foods):
        ax.annotate(label, (PC1[i], PC2[i]))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Foods')
    plt.savefig('PCA_q2b.png')
    #plt.show()

if __name__ == "__main__":
    plot_q2a()
    plot_q2b()
