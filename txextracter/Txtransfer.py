import pandas as pd
import numpy as np
from tqdm import tqdm
from time import sleep
from time import time
from collections import Counter
import tensorflow as tf
import datetime
import sys
import calendar

def count_tx(func):
    def wrapper(*args):
        df = func(*args)
        print(df[0], df[1].shape[0])
    return wrapper

def timer(func):
    def wrapper(*args):
        t1 = time()
        df = func(*args)
        print('Use time: ', time() - t1, 's')
    return wrapper

class Date:
    def __init__(self, dataname):
        '''
        dataname: file name (ex. Daily_2020_02_11)
        year: trading year (ex. 2020)
        month: trading month (ex. 2 = February)
        day: trading day (ex. 11)
        date: trading date (ex. 2020-02-11 00:00:00)
        weekday: day of the week (ex. 11 is Tuesday)
        mon_week: the number of days in current month (ex. Until today, Tuesday has appear twice in current month)
        quarter_month: Quarterly contract
            (url: https://www.taifex.com.tw/file/taifex/event/cht/NewthirdMonth/%E5%9C%8B%E5%85%A7%E8%82%A1%E5%83%B9%E6%8C%87%E6%95%B8%E6%9C%9F%E8%B2%A8%E5%88%B0%E6%9C%9F%E4%BA%A4%E5%89%B2%E6%9C%88%E4%BB%BD%E8%AA%BF%E6%95%B4_QA.pdf)
        deli_mon: delivery month contract (url please look above) 
        '''
        self.dataname = dataname
        self.year = int(dataname.split('_')[-3])
        self.month = int(dataname.split('_')[-2])
        self.day = int(dataname.split('_')[-1])
        self.date = datetime.datetime(self.year, self.month, self.day)
        self.weekday = datetime.datetime(self.year, self.month, self.day).weekday() + 1
        self.mon_week = self.date.isocalendar()[1] - self.date.replace(day = 1).isocalendar()[1] + 1
        
        self.quarter_month = [str(self.year) + '0' + str(i) if i < 10 else \
                    str(self.year) + str(i) for i in range(3, 13, 3)]
        self.deli_mon = []
        ### 
        calendar.setfirstweekday(calendar.SUNDAY)


    def cal_delimon(self):
        mon_len = len([1 for i in calendar.monthcalendar(self.year, self.month) \
            if i[self.weekday] != 0 and i[self.weekday] <= self.day])

        ### need improve
        if mon_len >= 3 and self.weekday >= 3:
            self.deli_mon = [str(self.year) + '0' + str(i) if i < 10 else \
                    str(self.year) + str(i) for i in range(self.month + 1, self.month + 3)]
        else:
            self.deli_mon = [str(self.year) + '0' + str(i) if i < 10 else \
                    str(self.year) + str(i) for i in range(self.month, self.month + 3)]

        ### Traverse the month of the entire quarter to confirm 
        ### the delivery month contract for that day
        ### ex. (202002 202003 202004 202006 202009 202012)
        self.deli_mon = self.deli_mon + [i for i in self.quarter_month if int(i[-2:]) > \
            int(self.deli_mon[-1][-2:])]

        ### total deliver month contract month must be number of six 
        if len(self.deli_mon) != 6:
            reg = [str(self.year + 1) + '0' + str(i) if i < 10 else \
                str(self.year + 1) + str(i) for i in range(6 - len(self.deli_mon))]
            self.deli_mon = self.deli_mon + reg

class Transfer(Date):
    def __init__(self, dataname, **kwargs):
        super().__init__(dataname)
        self.df = None
        self.eng = kwargs['engine']
        self.enc = kwargs['encode']
        self.com = kwargs['commodity']
        self.fil = kwargs['fil']
        self.exp = kwargs['exp_mon']
        self.cls = kwargs['cls_time']

    #def read_csv(dataname):
    def read_tx(self, **kwargs):   
        try:
            df = pd.read_csv(self.dataname + '.csv', sep = ',', engine = self.eng, \
                    encoding = self.enc)
            ### To remove white space everywhere for df.columns ###
            df.columns = df.columns.str.replace(' ', '')
        except Exception as e:
            print("read ", self.dataname, ".csv error:", sys.exc_info()[0])
        else:
            ### remove useless features from df ###
            for value in kwargs.values():
                if value in df.columns.values:
                    df = df.drop([value], 1)
            self.df = df

    @count_tx
    def fillt_(self):
        self.df = self.df[self.df[self.com].str.strip() == self.fil]
        self.df = self.df.reset_index(drop = True)
        self.df = self.df.drop(self.com, 1)
        return ('Filtered the Tx that totals are', self.df)

    @count_tx
    def fillt_nonDelimon(self):
        #print('filted Delivery month of non', calendar.month_name[self.month])
        self.cal_delimon()
        ### filter for months that are not in delivery month
        self.df = self.df[self.df[self.exp].str.strip().isin(self.deli_mon)]
        self.df = self.df.reset_index(drop = True)
        return ('Filtered in delivery month that totals are', self.df)

    @count_tx
    def fillt_nonGentrans(self):       
        ### filter the cls_time that if bigger equal than 84500 and smaller equal than 134500
        ### then that data be the General trading times data  
        test = [i for i, v in enumerate(self.df[self.cls]) if v >= 84500 and v <= 134500]
        self.df = self.df[self.df.index.isin(test)]
        self.df = self.df.reset_index(drop = True)
        return ('Filtered in General trading time that totals are', self.df)

    @timer
    def conversion(self):
        data = pd.DataFrame(columns = ["成交日期","交割月份","成交時間","開盤價","最高價","最低價","收盤價","成交量"])
        freqk = 100
        count = 0
        df_array = np.array(self.df)

        '''
        o = int(self.df.loc[0, "成交價格"])
        h = int(self.df.loc[0, "成交價格"])
        l = int(self.df.loc[0, "成交價格"])
        c = int(self.df.loc[0, "成交價格"])
        v = int(self.df.loc[0, "成交數量(B+S)"])
        self.data.loc[0, "成交時間"] = self.df.loc[0, "成交時間"]
        '''
        o = df_array[0, 3]
        h = df_array[0, 3]
        l = df_array[0, 3]
        c = df_array[0, 3]
        v = df_array[0, 4]
        data.loc[0, "成交時間"] = df_array[0, 2]
        print("The data of 開盤價:{0}, 最高價:{1}, 最低價:{2}, 收盤價:{3}, 成交量:{4}, 成交時間:{5}".format(o, h, l, c,\
         v, data.loc[0, "成交時間"]))

        for i in range(1, df_array.shape[0]):
            if (df_array[i, 2] // freqk) != (df_array[i - 1, 2] // freqk):
                print("The data of 開盤價:{0}, 最高價:{1}, 最低價:{2}, 收盤價:{3}, 成交量:{4}, 成交時間:{5}".format(o, h, l, c,\
                    v, df_array[i, 2]))
                data.loc[count, "開盤價"] = o
                data.loc[count, "最高價"] = h
                data.loc[count, "最低價"] = l
                data.loc[count, "收盤價"] = c
                data.loc[count, "成交量"] = v
                data.loc[count, "成交時間"] = df_array[i, 2]
                data.loc[count, "交割月份"] = df_array[i, 1]
                o = df_array[i, 3]
                h = df_array[i, 3]
                l = df_array[i, 3]
                c = df_array[i, 3]
                v = df_array[i, 4]
                count += 1
            else:
                if df_array[i, 3] > h:
                    h = df_array[i, 3]
                if df_array[i, 3] < l:
                    l = df_array[i, 3]
                c = df_array[i, 3]
                v = v + df_array[i, 4]                

        # data.loc[count, "開盤價"] = o
        # data.loc[count, "最高價"] = h
        # data.loc[count, "最低價"] = l
        # data.loc[count, "收盤價"] = c
        # data.loc[count, "成交量"] = v
        # data.loc[count, "成交時間"] = df_array[-1, 2]
        # data.loc[count, "交割月份"] = df_array[-1, 1]
        data.loc[:, "成交日期"] = df_array[0, 0]
        self.data = data
        self.data.to_csv('data\\' + self.dataname + '.csv', index = False, sep = ',', encoding='ANSI')
        return self.data

# TEST #
if __name__ == '__main__':
    # python Txtransfer.py --dataname=Daily_2020_02_11 --encode=big5
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('dataname', '', 'csv file name')
    tf.app.flags.DEFINE_string('exp_mon', '到期月份(週別)', 'Expiration month')
    tf.app.flags.DEFINE_string('cls_time', '成交時間', 'Closing time')
    tf.app.flags.DEFINE_string('recent', '近月價格', 'recent prices value')
    tf.app.flags.DEFINE_string('far', '遠月價格', 'far month prices value')
    tf.app.flags.DEFINE_string('open', '開盤集合競價', 'open auction value')
    tf.app.flags.DEFINE_string('Commodity', '商品代號', 'Commodity code to filt columns')
    tf.app.flags.DEFINE_string('fil', 'TX', 'field to select')
    tf.app.flags.DEFINE_string('engine', 'python', 'read csv engine')
    tf.app.flags.DEFINE_string('encode', 'utf-8', 'read csv encoding(default utf-8)')

    trans = Transfer(
        FLAGS.dataname, 
        engine = FLAGS.engine,
        encode = FLAGS.encode,
        commodity = FLAGS.Commodity,
        fil = FLAGS.fil,
        exp_mon = FLAGS.exp_mon,
        cls_time = FLAGS.cls_time,)

    trans.read_tx(
        recent = FLAGS.recent,
        far = FLAGS.far,
        open = FLAGS.open
    )

    trans.fillt_()
    trans.fillt_nonDelimon()
    trans.fillt_nonGentrans()
    trans.conversion()
    # df = read_csv('Daily_2018_12_22') 
    # dfcor, count = fillter_nonTX(df)
    # datacor, count = fillter_nonDeliverymonth(dfcor, count, '201901')
    # dataans, count = fillter_nonGeneraltransact(datacor, count)
    # ans = conversion(dataans, count, 'Daily_2018_12_22')




