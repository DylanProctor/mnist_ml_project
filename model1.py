#Need scipy, scikit-learn, need tensorflow to use keras, need keras
from numpy import mean
from numpy import std 
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
import time

#Model 1 notes:
#So for the first test of the model with no improvements and basline cnn 
#we got a mean of 98.667 for the n = 5 fold validation score and a std = 0.166


#Model 2 notes:
#With the added batch normalization 
#we have a mean of 98.710 for n = 5 fold validation and a std = 0.085
#So it doesn't seem to offer much benefit and seems to take longer to compute

#Model 3/4 notes:
#Next I am going to remove the batch normalization and try to change the learning rate
#I will have two trials with a learning rate of 0.001 and then 0.0001 for model 3 and 4 respectively
#Model 3 got mean = 97.882 and std = 0.267 with n = 5 fold validation. It took 787 sec
#It seems that we have a lower mean score and higher variance as one can see from the box plot and std
#Model 4 got mean = 94.630, std = 0.229 with n = 5 fold validation. It took 678seconds.
#It is even worse in the avrage score than the model with a lr of 0.001

#Model 5:
#Since the scores got worse with a decrease of lr lets try to see if there is any improvement with a lr of 0.1
#It got a mean = 96.252 with std = 0.625, and it took 694 sec
#The plot of the diagnostics curves had a lot of bouncing back and forth in its scores compared to the more
#gradual ascent/descent of the curve. This can also be attributed by it unsually high std suggesting a high variance
#It seems there is a sweet spot for the learning rate and it looks like 0.01 seems good for now

#Model 6:
#This last model will introduce a method that is very commonly used to improve models which is to simply add
#depth to the ccn. There is a joke among deep learning engineers that all you need to do is add more layers 
#to improve a model. This done by adding more convolutional and pooling layers with the same filter size while
#increasing the number of filters. So I am going to add 2 more conv layers with now 64 filters and then another 
#max pooling layer
#It got mean = 98.965 with std = 0.163, and it took 2076 sec
#So it got an increase in the average score with a high up to 99.258 which I think it is pretty good
#The increase in layers also certainly made it take much longer to train and validate

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm/255.0
    test_norm = test_norm/255.0

    return train_norm, test_norm

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (28, 28, 1)))
    # for model 2 we will add batch normalization after the conv and connected layers
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    #For model 6 I added these 2 conv layers and the max pooling layer to improve accuracy
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer = 'he_uniform'))
    # for model 2 we add batch normalization
    #model.add(BatchNormalization())
    model.add(Dense(10, activation = 'softmax'))
    #Model 1 had a lr of 0.01, model 3 will have a lr of 0.001, model 4 will have 0.0001, and model 5 will have 0.1
    opt = SGD(learning_rate = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model


def evaluate_model(dataX, dataY, n_folds = 5):
    scores, histories = list(), list()

    kfold = KFold(n_folds, shuffle = True, random_state = 1)

    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs = 10, batch_size = 32, validation_data = (testX, testY), verbose = 0)
        _, acc = model.evaluate(testX, testY, verbose = 0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)

    return scores, histories

def summarize_diagonstics(histories):
    for i in range(len(histories)):
        plt.subplot(2,1,1)
        plt.title('Cross Entropy loss')
        plt.plot(histories[i].history['loss'], color = 'blue', label = 'train')
        plt.plot(histories[i].history['val_loss'], color = 'orange', label = 'test')

        plt.subplot(2,1,2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color = 'blue', label = 'train')
        plt.plot(histories[i].history['val_accuracy'], color = 'orange', label = 'test')
    plt.show()

def summarize_performance(scores):
    print('Accuracy: mean = %.3f std = %.3f, n = %d' % (mean(scores)*100, std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()

def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)
    summarize_diagonstics(histories)
    summarize_performance(scores)

t0 = time.perf_counter()

run_test_harness()

tf = time.perf_counter() - t0

print('It took %i seconds' % tf)