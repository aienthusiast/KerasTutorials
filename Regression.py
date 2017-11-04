from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt


bostonData = load_boston()

data, target = bostonData['data'], bostonData['target']

trainData, testData, trainTarget, testTarget = train_test_split(
        data, target,
        test_size = .33
        )

networkShape = [data.shape[1], 32, 64, 48, 1]

model = Sequential()

for i in range(len(networkShape)-2):
    
    model.add(Dense(networkShape[i+1], input_dim = networkShape[i], activation = 'relu'))

model.add(Dense(networkShape[-1], input_dim = networkShape[-2]))

model.compile(loss = 'mse', optimizer = 'rmsprop')

stepSize = 10
x = range(stepSize, 1000, stepSize)
trainScores = []
testScores = []

for _ in x:
    
    model.fit(trainData, trainTarget, epochs = stepSize, batch_size = 32)
    
    trainScore = model.evaluate(trainData, trainTarget, batch_size = 64)
    testScore = model.evaluate(testData, testTarget, batch_size = 64)
    
    testScores.append(testScore)
    trainScores.append(trainScore)

testLine = plt.plot(x, testScores, 'r')
trainLine = plt.plot(x, trainScores, 'b' )

plt.show()








#bostonData = load_boston()
#
#
#data, target = bostonData['data'], bostonData['target']
#
#trainData, testData, trainTarget, testTarget = train_test_split(
#        data, target, test_size=0.33)
#
#
#print(data.shape)
#
#
#model = Sequential()
#model.add(Dense(1, input_dim = data.shape[1]))
#model.compile(loss='mse', optimizer='rmsprop')
#
#model.fit(trainData, trainTarget, epochs=1000, batch_size=16,verbose=0)
##model.fit(X_train, y_train, nb_epoch=1, batch_size=16,verbose=1)
#trainScore = model.evaluate(trainData, trainTarget, batch_size=16)
#testScore = model.evaluate(testData, testTarget, batch_size=16)
#print(testScore, trainScore)