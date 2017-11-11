import datetime

#data week1-week32 predict week33
#user_id*27time_slot

for i in range(1, 2):
    data_file_path = '/data/public/data-%03d.csv'%(i);
    with open(data_file_path) as data_file:
        data_file.readline()
        for data_line in data_file:
            data = data_line.split(',')
            user_id = data[0]
            datetime_object = datetime.datetime.strptime(data[4], '%Y-%m-%d %H:%M:%S.%f')
            week_of_yaer = datetime_object.isocalendar()[1]
            weekday = datetime_object.weekday()
            hour = datetime_object.hour
