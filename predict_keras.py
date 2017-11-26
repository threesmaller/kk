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

def output_result(predict):
    solution = np.asarray(predict)
    fd = open('/data/solution.csv', 'w')
    fd.write('user_id,time_slot_0,time_slot_1,time_slot_2,time_slot_3,time_slot_4,time_slot_5,time_slot_6,time_slot_7,time_slot_8,time_slot_9,time_slot_10,time_slot_11,time_slot_12,time_slot_13,time_slot_14,time_slot_15,time_slot_16,time_slot_17,time_slot_18,time_slot_19,time_slot_20,time_slot_21,time_slot_22,time_slot_23,time_slot_24,time_slot_25,time_slot_26,time_slot_27')
    fd.write('\n')
    for i in range (0, 37092):
        fd.write(str(i + 57159))
        for j in range (0, 28):
            fd.write(',' + str.format("{0:.3f}", solution[i, j]));
        fd.write('\n')
    fd.close()

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

def load_user(job_id):
    global data_matrix
    global target_matrix
    global real_data_matrix
    global samples
    global tests
    global jobs

    iter = (samples + tests) / jobs
    for i in range(0, iter):
        user_id = i * jobs + job_id
        user_file_path = '/data/user/data-%03d.pkl'%(user_id)
        if not os.path.isfile(user_file_path):
            return
        matrix = cPickle.load(open(user_file_path, 'rb'))
        if user_id < samples:
            data_matrix[user_id, :, :] = matrix[0, 0:32, :]
            target_matrix[user_id, :, :] = matrix[0, 32, :]
        else:
            real_data_matrix[user_id - samples, :, :] = matrix[0, 0:32, :]

class loadThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        load_user(self.threadID)

def load():
    for i in range(0, jobs):
        threads = []
        thread = loadThread(i)
        thread.start()
        threads.append(thread)
    for t in threads:
        t.join()

samples = 57159
#tests = 37092
tests = 1
output = 0
jobs = 200
data_matrix = np.zeros((samples, 32, 28))
target_matrix = np.zeros((samples, 1, 28))
real_data_matrix = np.zeros((tests, 32, 28))

load()

x_train = data_matrix[0:samples, :, :].reshape((samples, -1))
y_train = target_matrix[0:samples, :, :].reshape((samples, -1))
for i in range(0, y_train.shape[0]):
    for j in range(0, y_train.shape[1]):
        if y_train[i, j] > 0:
            y_train[i, j] = 1
n_classes = y_train.shape[1]

x_test = real_data_matrix[0:tests, :, :].reshape((tests, -1))

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

model.fit(x_train, y_train, epochs = 20, batch_size = 1000, validation_split = 0.2)

y_score = model.predict(x_test, batch_size = 1000, verbose = 0)
if output == 1:
    output_result(y_score)
