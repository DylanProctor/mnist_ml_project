from tensorflow.keras.datasets import mnist
from matplotlib import pyplot

(trainX, trainY), (testX, testY) = mnist.load_data()

print('Train: X = %s, Y = %s' % (trainX.shape, trainY.shape))
print('Test: X = %s, Y = %s' % (testX.shape, testY.shape))

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(trainX[i], cmap = pyplot.get_cmap('gray'))

pyplot.show()