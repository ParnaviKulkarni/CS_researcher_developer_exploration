# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:41:44 2024

@author: Parnavi Kulkarni
"""

# -*- coding: utf-8 -*-
"""
Code & Mathematics Explanation
"""
# digit classification using neural netwroks from scratch
# Math
# 28 * 28 pixels images
# each pixel has values from 0-255
# each image= 1 row of 28*28=784 values, matrix=list of rows of corresponding image values
# in calculation, we take transpose of the matrix, each column=one example
# associate each column to a value from 0-9(outputs(digits))
# 3 layer neural n/w
# input layer(784), 1 hidden layer, 1 output layer

# forward propagation
# z1 --> unacctivated 1st layer, w0--> weights, a0--> inputs, b --> bias
# A0 = X (784 * m)
# Z1 (10 * m) = W1A0 (10*784 & 784*m) + b0 (10*m)
# A1 = g(Z1) = ReLU(Z1) <-- Rectified Linear Unit Activation function =x if x>0 else 0, if x<=0
# Z2 (10 * m) = W2A1 (10*10 & 10*m) + b1 (10*m)
# A2 = softmax(Z2) <-- probabilities for all possible outputs, e^zi/sumof(e^zj) (all the nodes)

# back propagation
# dZ2 = A2(actual label) - Y (one hot encoded output)
# dW2 = (1/m) * dZ2A1T (a1 transpose)  (m=output size)
# db2 = (1/m) * sumof(dZ2)
# dZ1 = W2T dZ2 * g'(Z1) <-- derivation of activation function, undo activation)
## Weights updation (alpha=learning rate)
# W1 = W1(old) - alpha*dW1
# b1 = b1(old) - alpha*db1
# W2 = W2(old) - alpha*dW2
# b2 = b2(old) - alpha*db2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_train = pd.read_csv('C:/Users/abcom.in/Downloads/digit-recognizer/train.csv')
data_test = pd.read_csv('C:/Users/abcom.in/Downloads/digit-recognizer/train.csv')

#print(data.head())

data_train = np.array(data_train)
m,n = data_train.shape
np.random.shuffle(data_train)



data_train = data_train.T 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

data_test = np.array(data_test)
m1,n1 = data_test.shape
data_test = data_test.T
Y_test = data_test[0]
X_test = data_test[1:n1]
X_test = X_test / 255.

def init_parameters():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    softMax = np.exp(Z) / sum(np.exp(Z))
    #print(softMax)
    return softMax

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    #print(Z1)
    A1 = ReLU(Z1)
    #print(A1)
    Z2 = W2.dot(A1) + b2
    #print(Z2)
    A2 = softmax(Z2)
    #print(A2)
    return Z1, A1, Z2, A2

def one_hot_encoded(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def derivatived_ReLU(Z):
    return Z>0

def backward_propagation(Z1, A1, Z2, A2, W2,X, Y):
    m = Y.size
    one_hot_Y = one_hot_encoded(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * derivatived_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y, iterations, alpha):
    W1, b1, W2, b2 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2,A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ",i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# Training
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 501, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(X, Y, index, W1, b1, W2, b2):
    current_image = X[:,index, None]
    prediction = make_predictions(X[:,index,None], W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Testing / Validating
print("Validation")
test_predictions = make_predictions(X_test, W1, b1, W2, b2)
test_accuracy=get_accuracy(test_predictions, Y_test)
print(test_accuracy)
test_prediction(X_test, Y_test, 0, W1, b1, W2, b2)
test_prediction(X_test, Y_test, 2, W1, b1, W2, b2)
test_prediction(X_test, Y_test, 10, W1, b1, W2, b2)
    