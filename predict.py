import numpy, cPickle, os.path, random, math
from sklearn import metrics, svm
import output as kk_output
import score as kk_score
import parse as kk_parse
import matplotlib.pyplot as plt

def training(threshold):
    predict = numpy.zeros((94251, 28))
    create_week = kk_parse.parse_create_time()
    for user_id in range(1, 94251):
        user_file_path = '/data/user/data-%03d.pkl'%(user_id);
        if not os.path.isfile(user_file_path):
            continue
        data_matrix = cPickle.load(open(user_file_path, 'rb'))
        matrix_predict = numpy.zeros((1, 1, 28))
        matrix_predict_raw = numpy.zeros((1, 1, 28))
        for i in range(0, 32):
            for j in range(0, 28):
                matrix_predict_raw[0, 0, j] += math.fabs(i*math.log(data_matrix[0, i, j]+0.00001))
        for j in range(0, 28):
            matrix_predict_raw[0, 0, j] *= -1
        matrix_predict = matrix_predict_raw.reshape(-i)
        predict[user_id, :] = matrix_predict
    return predict

def draw(fpr, tpr, score):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % score, marker='o')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

max_score = 0
for threshold in [0.9]:
    predict = training(threshold)
    fpr, tpr, score = kk_score.train_score(predict[0:57159, :])
    draw(fpr, tpr, score)    
    print(threshold, score)
    if score >= max_score:
        kk_output.output_result(predict)
