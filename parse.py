import os, datetime, numpy, fractions, time, threading, cPickle

#train:week1-week32 predict:week33
#week * user_id * time_slot

#weekday Mon:0 Sun:6
#slot_0:1-9
#slot_1:9-17
#slot_2:17-21
#slot_3:21-24,0-1
#slot:0-27

#user_id:0-94250
#train:0-57158 predict:57159-94250

#(user_id, week, slot)
#data_matrix = numpy.zeros((94251, 53, 29))

def parse_data_file(file_id):
    global data_matrix
    data_file_path = '/data/public/data-%03d.csv'%(file_id);
    with open(data_file_path) as data_file:
        data_file.readline()
        for data_line in data_file:
            data = data_line.split(',')
            user_id = data[0]
            datetime_object = datetime.datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S.%f')
            week = datetime_object.isocalendar()[1]
            weekday = datetime_object.weekday()
            hour = datetime_object.hour
            slot_offset = weekday * 4
            if 1 < hour <= 9:
                slot = 0
                slot_size = 8*60*60
            elif 9 < hour <= 17:
                slot = 1
                slot_size = 8*60*60
            elif 17 < hour <= 21:
                slot = 2
                slot_size = 4*60*60
            else:
                slot = 3
                slot_size = 4*60*60
            slot += slot_offset
            play_duration = data[5]
            x = int(user_id) + 1
            y = week
            z = slot + 1
            value = float(play_duration) / slot_size
            data_matrix[x, y, z] += value

class myThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        parse_data_file(self.threadID)

pkl_file_path = '/data/data_duration.pkl'
if os.path.exists(pkl_file_path):
    data_matrix = cPickle.load(open('/data/data_duration.pkl', 'rb'))
else:
    data_matrix = numpy.zeros((94251, 53, 29))
    numpy.set_printoptions(precision = 4)

    threads = []
    for i in range(1, 76):
        thread = myThread(i)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    cPickle.dump(data_matrix, open('/dat/data_duration.pkl', 'wb'))

print(data_matrix[1, 23, 14])
print(1, 23, 14, 0.238645833333)
