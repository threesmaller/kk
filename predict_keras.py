import os, cPickle, threading, random, datetime
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K

def binary_PFA(y_true, y_pred, threshold=K.variable(value = 0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value = 0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)    
    return TP/P
    
def keras_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred,k) for k in np.linspace(0, 1, 1000)], axis = 0)
    pfas = tf.stack([binary_PFA(y_true, y_pred,k) for k in np.linspace(0, 1, 1000)], axis = 0)
    pfas = tf.concat([tf.ones((1,)) , pfas], axis = 0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis = 0)

def load_user(user_id):
    global data_matrix
    global target_matrix
    user_file_path = '/data/user/data-%03d.pkl'%(user_id)
    if not os.path.isfile(user_file_path):
        return
    matrix = cPickle.load(open(user_file_path, 'rb'))
    data_matrix[user_id, :, :] = matrix[0, 0:32, :]
    target_matrix[user_id, :, :] = matrix[0, 32, :]

class loadThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        load_user(self.threadID)

def load(samples, jobs):
    iters = samples / jobs
    for j in range(0, iters):
        threads = []
        for i in range(0, jobs):
            thread = loadThread(j*jobs + i)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()


samples = 50000
jobs = 100
data_matrix = np.zeros((samples, 32, 28))
target_matrix = np.zeros((samples, 1, 28))

load(samples, jobs)

X = data_matrix[0:samples, :, :].reshape((samples, -1))
y = target_matrix[0:samples, :, :].reshape((samples, -1))
for i in range(0, y.shape[0]):
    for j in range(0, y.shape[1]):
        if y[i, j] > 0:
            y[i, j] = 1
n_classes = y.shape[1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.4)

scaler = MinMaxScaler()
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)  

model = Sequential()
model.add(Dense(512, activation='relu', input_dim = x_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[keras_auc])

model.fit(x_train, y_train, epochs = 20, batch_size = 500, validation_split = 0.2)

y_score = model.predict(x_test, batch_size = 500, verbose = 0)
fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)
print('score:' + str(roc_auc))
