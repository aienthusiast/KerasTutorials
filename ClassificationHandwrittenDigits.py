from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

networkShape = [784, 32, 64, 48, 10]

def buildModel(networkShape):

    model = Sequential()
    
    for i in range(len(networkShape)-2):
        
        model.add(Dense(networkShape[i+1], input_dim = networkShape[i], activation = 'relu'))
    
    model.add(Dense(networkShape[-1], input_dim = networkShape[-2], activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')
    
    return model

def plotProgress(model, trainDataset, testDataset, stepSize = 1, numEpochs = 10):

    trainData, trainTarget = trainDataset
    testData, testTarget = testDataset
    
    x = range(stepSize, numEpochs, stepSize)
    trainScores = []
    testScores = []
    
    for _ in x:
        
        model.fit(trainData, trainTarget, epochs = stepSize)
        
        trainScore = model.evaluate(trainData, trainTarget)
        testScore = model.evaluate(testData, testTarget)
        
        testScores.append(testScore)
        trainScores.append(trainScore)
    
    plt.plot(x, testScores, 'r')
    plt.plot(x, trainScores, 'b' )
    
    plt.show()
    
model = buildModel(networkShape)
plotProgress(model, 
             [mnist.train.images, mnist.train.labels],
             [mnist.test.images, mnist.test.labels]
             )



