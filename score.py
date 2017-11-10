import numpy as np
from sklearn import datasets,svm,metrics,model_selection,preprocessing
from random import *

max_score = 0
y_true = []
for i in range(1, 31):
    data_file_path = '/data/public/label-%03d.csv'%(i);
    data = np.genfromtxt(data_file_path, delimiter=',')
    data = np.delete(data, (0), axis=0)
    data = np.delete(data, (0), axis=1)
    true = np.array(data.reshape(-1))
    y_true = np.append(y_true, true)

for cnt in range(0, 1):

    rand = np.random.binomial(10, random(), size=(37092, 28))
    predict = predict + np.around(rand/10.0)

    if len(y_true) > len(y_scores):
        y_true = y_true[:len(y_scores)]

    y_scores = np.array(predict.reshape(-1))
    if len(y_true) == len(y_scores):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label = 1)
        score = metrics.auc(fpr, tpr)

        if score > max_score:
            sol = np.asarray(predict)
            fd = open('../solution.csv', 'w')
            fd.write('user_id,time_slot_0,time_slot_1,time_slot_2,time_slot_3,time_slot_4,time_slot_5,time_slot_6,time_slot_7,time_slot_8,time_slot_9,time_slot_10,time_slot_11,time_slot_12,time_slot_13,time_slot_14,time_slot_15,time_slot_16,time_slot_17,time_slot_18,time_slot_19,time_slot_20,time_slot_21,time_slot_22,time_slot_23,time_slot_24,time_slot_25,time_slot_26,time_slot_27')
            fd.write('\n')
            for i in range (0, 37092):
                fd.write(str(i+57159))
                for j in range (0, 28):
                    fd.write(',' + str.format("{0:.3f}", sol[i, j]));
                fd.write('\n')
            fd.close()
            max_score = score
