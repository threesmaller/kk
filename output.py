import os, numpy, cPickle

def output_result(predict):
    solution = numpy.asarray(predict)
    fd = open('/data/solution.csv', 'w')
    fd.write('user_id,time_slot_0,time_slot_1,time_slot_2,time_slot_3,time_slot_4,time_slot_5,time_slot_6,time_slot_7,time_slot_8,time_slot_9,time_slot_10,time_slot_11,time_slot_12,time_slot_13,time_slot_14,time_slot_15,time_slot_16,time_slot_17,time_slot_18,time_slot_19,time_slot_20,time_slot_21,time_slot_22,time_slot_23,time_slot_24,time_slot_25,time_slot_26,time_slot_27')
    fd.write('\n')
    for i in range (57159, 94251):
        fd.write(str(i))
        for j in range (0, 28):
            fd.write(',' + str.format("{0:.3f}", solution[i, j]));
        fd.write('\n')
    fd.close()
