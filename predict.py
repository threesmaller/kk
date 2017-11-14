import numpy, cPickle, os.path
from sklearn import metrics

sqr_error = 0
total_data = 0

numpy.set_printoptions(threshold = numpy.nan, precision = 4, suppress = True)
for user_id in range(1, 57159):
#for user_id in range(57159, 94251):
    user_file_path = '/data/user/data-%03d.pkl'%(user_id);
    if not os.path.isfile(user_file_path):
        continue
    data_matrix = cPickle.load(open(user_file_path, 'rb'))
    matrix_predict = numpy.zeros((1, 1, 28))
    matrix_predict_raw = numpy.zeros((1, 1, 28))
    for i in range(0, 32):
        matrix_predict_raw += data_matrix[0, i, :]*(i+1)
    matrix_predict = matrix_predict_raw.reshape(-i)/32
    for j in range(0, 28):
        if matrix_predict[j] > 1:
            matrix_predict[j] = 1
        else:
            matrix_predict[j] = matrix_predict[j]
    if user_id < 57159:
        matrix_real = data_matrix[0, 32, :].reshape(-i)
        for j in range(0, 28):
            if matrix_real[j] > 0:
                matrix_real[j] = 1
        sqr_error += metrics.mean_squared_error(matrix_predict, matrix_real)
        total_data += 1
    user_file_path = '/data/predict/data-%03d.pkl'%(user_id);
    cPickle.dump(matrix_predict, open(user_file_path, 'wb'))

if total_data > 0:
    print(sqr_error/total_data)
