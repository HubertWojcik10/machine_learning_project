'''
This is our implementation of a Neural Network model from scratch.
'''
import numpy as np
import pandas as pd

class MiniBatchGD:
    '''
        A class used to sample a batch of data from the dataset
    '''
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def sample(self):
        ''' Sample a batch of data '''
        idx = np.random.choice(self.X.shape[0], self.batch_size, replace=False)
        return self.X[idx], self.y[idx]

class ActivationFunction:
    '''
        A class used to define and calculate the activation function used in the network. 
        It enables us to easily change the activation function used in the network.
    '''
    def __init__(self, name, lr=0.01):
        self.name = name
        self.lr = lr

    def calculate(self, x, derivative=False):
        if self.name == 'sigmoid':
            return self.sigmoid(x, derivative=derivative)
        elif self.name == 'relu':
            return self.relu(x, derivative=derivative)
        elif self.name == 'softmax':
            return self.softmax(x, derivative=derivative)

    def sigmoid(self, x, derivative=False):
        ''' 
            sigmoid activation function and its derivative 
        '''
        if not derivative:
            return 1 / (1 + np.exp (-x))
        else:
            out = self.sigmoid(x)
            return out * (1 - out)

    def relu(self, x, derivative=False):
        ''' 
            relu activation function and its derivative 
        '''
        if not derivative:
            return np.where(x > 0, x, 0)
        else:
            return np.where(x > 0, 1, 0)
    
    def softmax(self, x, derivative=False):
        ''' 
            softmax activation function and its derivative 
        '''
        if not derivative:
            exps = np.exp(x - np.max(x))
            return exps / np.sum(exps)
        else:
            out = self.softmax(x)
            return out * (1 - out)


class NeuralNetwork:
    '''
        Our main class used to define the Neural Network model
    '''
    def __init__(self, input_size=784, hidden_size=300, output_size=5, layers_num=3, learning_rate=0.01, test=False, activation_name='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size 
        self.layers_num = layers_num
        self.lr = learning_rate
        self.activation = ActivationFunction(activation_name, lr=learning_rate)

        self.weights = []
        self.bias = []

        self.weights.append(np.random.randn(self.input_size, self.hidden_size))
        self.bias.append(np.random.randn(1, self.hidden_size))

        self.weights.append(np.random.randn(self.hidden_size, self.output_size))
        self.bias.append(np.random.randn(1, self.output_size))
        
    def forward_pass(self, X):
        ''' 
            conduct the forward pass on the network 
        '''
        #X = X / 255
        self.z1 = np.dot(X, self.weights[0]) + self.bias[0]
        self.a1 = self.activation.calculate(self.z1)

        self.z2 = np.dot(self.a1, self.weights[1]) + self.bias[1]
        self.a2 = self.activation.calculate(self.z2)

        self.outputs = np.zeros((len(self.a2), self.output_size))
        for i in range(len(self.a2)):
            self.outputs[i][np.argmax(self.a2[i])] = 1

        self.outputs = np.array(self.outputs)


    def backward_pass(self, X, y):
        '''
            conduct the backward pass on the network
        '''
        #X = X / 255
        y_mtrix = np.zeros((len(y), int(self.output_size))) 
        #change y into 1-hot encoding by assigning 1 to the index of the label
        for i in range(len(y)):
            y_mtrix[i][y[i]] = 1

        #loss, used to check the accuracy of the network
        self.loss = np.sum((self.outputs - y_mtrix)**2) / (2*y_mtrix.size)

        #accuracy, used to check the accuracy of the network
        self.accuracy = np.sum(np.argmax(self.outputs, axis=1) == y) / len(y)

        #calculate the error of the hidden layer
        self.e1 = self.a2 - y_mtrix
        dw1 = self.e1 * self.activation.calculate(self.a2, True)
        
        #calculate the error of the input layer
        self.e2 = np.dot(dw1, self.weights[1].T)
        dw2 = self.e2 * self.activation.calculate(self.a1, True)

        #update the weights
        w2_update = np.dot(self.a1.T, dw1) / len(X)
        w1_update = np.dot(X.T, dw2) / len(X)

        #update the biases
        b2_update = self.lr * np.sum(dw1, axis=0, keepdims=True) / len(X)
        b1_update = self.lr * np.sum(dw2, axis=0, keepdims=True) / len(X) 

        self.weights[1] -= self.lr * w2_update
        self.weights[0] -= self.lr * w1_update

        self.bias[1] -= self.lr * b2_update
        self.bias[0] -= self.lr * b1_update

        
    def TRAIN(self, X, y, epochs=5, testing=False):
        '''
            train the network for a given number of epochs
        '''
        l = list()
        

        for epoch in range(epochs):
            X_sample, y_sample = MiniBatchGD(X, y, batch_size=128).sample()
            self.forward_pass(X_sample)
            self.backward_pass(X_sample, y_sample)
            if testing: print(f'Epoch {epoch}, loss: {self.loss}, accuracy: {self.accuracy}')
            l.append([epoch, self.accuracy,self.loss])
        
        df = pd.DataFrame(data= l, columns=['epoch', 'accuracy', 'loss'])
        return df

    def TEST(self, X, y):
        '''
            test the network
        '''
        self.forward_pass(X)
        self.backward_pass(X, y)
        print(f'loss: {self.loss}, accuracy: {self.accuracy}')

'''if __name__ == '__main__':
    #load the data
    X_train, y_train, X_test, y_test = load_data()
    #create the network
    nn = NeuralNetwork(activation_name='relu')
    #train the network
    nn.TRAIN(X_train, y_train, epochs=50, testing=True)
    #test the network
    nn.TEST(X_test, y_test)'''