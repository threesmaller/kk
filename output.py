import os, numpy, cPickle

predict = numpy.zeros((37092, 28))

for user_id in range (57159, 94251):
    user_file_path = '/data/predict/data-%03d.pkl'%(user_id);
    if not os.path.isfile(user_file_path):
        continue
    data_matrix = cPickle.load(open(user_file_path, 'rb'))
    predict[user_id - 57159, :] = data_matrix

solution = numpy.asarray(predict)
fd = open('/data/solution.csv', 'w')
fd.write('user_id,time_slot_0,time_slot_1,time_slot_2,time_slot_3,time_slot_4,time_slot_5,time_slot_6,time_slot_7,time_slot_8,time_slot_9,time_slot_10,time_slot_11,time_slot_12,time_slot_13,time_slot_14,time_slot_15,time_slot_16,time_slot_17,time_slot_18,time_slot_19,time_slot_20,time_slot_21,time_slot_22,time_slot_23,time_slot_24,time_slot_25,time_slot_26,time_slot_27')
fd.write('\n')
for i in range (0, 37092):
    fd.write(str(i + 57159))
    for j in range (0, 28):
        fd.write(',' + str.format("{0:.3f}", solution[i, j]));
    fd.write('\n')
fd.close()

