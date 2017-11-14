import numpy, cPickle, os.path
from sklearn import metrics, svm
import output as kk_output
import score as kk_score
import parse as kk_parse

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
            matrix_predict_raw += data_matrix[0, i, :]*(i+1)
        matrix_predict = matrix_predict_raw.reshape(-i)/create_week[user_id]
        for j in range(0, 28):
            if matrix_predict[j] > threshold:
                matrix_predict[j] = 1
            else:
                matrix_predict[j] = matrix_predict[j]
        predict[user_id, :] = matrix_predict
    return predict

max_score = 0
for threshold in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]:
    predict = training(threshold)
    score = kk_score.train_score(predict[0:57159, :])
    print(threshold, score)
    if score >= max_score:
        kk_output.output_result(predict)
