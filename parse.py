import datetime

#train:week1-week32 predict:week33
#week * user_id * time_slot

#weekday Mon:0 Sun:6
#slot_0:1-9
#slot_1:9-17
#slot_2:17-21
#slot_3:21-24,0-1

for i in range(1, 2):
    data_file_path = '/data/public/data-%03d.csv'%(i);
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
            elif 9 < hour <= 17:
                slot = 1
            elif 17 < hour <= 21:
                slot = 2
            else:
                slot = 3
            slot += slot_offset
#(week, user_id, slot)
