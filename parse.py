import os, datetime, numpy, fractions, time, threading, cPickle, random

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
#data_matrix = numpy.zeros((94252, 53, 29))

def parse_data_file(file_id):
    last_user_id = -1
    data_file_path = '/data/public/data-%03d.csv'%(file_id);
    data_matrix = numpy.zeros((1, 53, 28))
    with open(data_file_path) as data_file:
        data_file.readline()
        for data_line in data_file:
            data = data_line.split(',')
            user_id = int(data[0])
            user_check_file_path = '/data/user/data-%03d.pkl'%(user_id);
            if os.path.isfile(user_check_file_path):
                continue
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
     
            if last_user_id == -1:
                data_matrix = numpy.zeros((1, 53, 28))
                user_file_path = '/data/user/data-%03d.pkl'%(user_id);
                last_user_id = user_id
            elif last_user_id != user_id:
                cPickle.dump(data_matrix, open(user_file_path, 'wb'))
                user_file_path = '/data/user/data-%03d.pkl'%(user_id);
                data_matrix = numpy.zeros((1, 52, 28))
                last_user_id = user_id
            x = 0
            y = week - 1
            z = slot
            value = float(play_duration) / slot_size
            data_matrix[x, y, z] += value
        cPickle.dump(data_matrix, open(user_file_path, 'wb'))

def parse_create_time():
    user_week = {}
    data_file_path = '/data/public/user_create_time.csv'
    with open(data_file_path) as data_file:
        data_file.readline()
        for data_line in data_file:
            data = data_line.split(',')
            user_id = int(data[0])
            create_day = data[1].rstrip() + '-01'
            datetime_object = datetime.datetime.strptime(create_day, '%Y-%m-%d')
            if datetime_object.year < 2017:
                week = 1
            else:
                week = datetime_object.isocalendar()[1]
                if week == 52:
                    week = 1
            total_week = 33 - week
            user_week[user_id] = total_week
    return user_week

class myThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        parse_data_file(self.threadID)

def parse():
    threads = []
    for i in range(1, 76):
        thread = myThread(i)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
