import pandas as pd
import numpy as np
#import 

def read_csv(dataname):
    print("資料讀取中.....")
    df = pd.read_csv(dataname + '.csv', sep = ',',engine='python')
    print("資料讀取完畢!")
    #time.sleep(5)
    print("刪除資料中.....")
    df = df.drop(['近月價格'], 1)
    df = df.drop(['遠月價格'], 1)
    df = df.drop(['開盤集合競價 '], 1)
    print("刪除資料完畢!")
    #time.sleep(10)
    return df

def fillter_nonTX(df):
    df = np.array(df)
    dfcor = df
    count = 0
    for i in range(0, len(df)):
        #如果找不到TX這個商品代號的話就捨去
        #如果找到有TX這個商品但是第一個字是M的話也捨去(例:MTX)
        if (df[i, 1].find('TX') == -1) or (df[i, 1].find('M') == 0):
            print("非TX，過濾掉第", i, "筆資料...")
        else:
            #符合的話把TX丟到新的array
            #計數器 + 1
            dfcor[count] = df[i]
            count = count + 1
    return dfcor, count
#count = 101235
def fillter_nonDeliverymonth(dfcor, count, keyword):
    #計數器歸0
    countTX = count 
    count = 0
    datacor = dfcor
    print("過濾掉非11月的交割月份...")
    for i in range(0, countTX):
        #二次過濾資料，過濾掉非9月的資料
        #如果到期月份也就是交割的月份不是9月的話就捨去
        if (datacor[i, 2].find(keyword) == 0 and (datacor[i, 2].find('/') == -1)):
            datacor[count] = dfcor[i]
            count = count + 1
            #print("count's: ",count)
        else:
            print("非11月，過濾掉第", i, "筆資料...")
            #datacor = np.delete(datacor ,count, axis = 0)
    print("二次過濾完畢!")
    return datacor, count
def fillter_nonGeneraltransact(datacor, count):
    dataans = datacor
    countTX = count
    count = 0
    print("過濾掉非一般的交易時段中.....")
    for i in range(0, countTX):
        time = str(datacor[i, 3])
        #如果時間的長度是6的話代表會有小時、分以及秒
        if len(str(datacor[i, 3])) == 6:        
            Hr = int(time[0:2])
            Min = int(time[2:4])
            Sec = int(time[4:6])
        #如果時間的長度是2的話代表只有秒
        elif len(str(datacor[i, 3])) == 2:
            Hr = 0
            Min = 0
            Sec = int(time[0:2])
        #如果時間的長度是3的話代表有分和秒
        elif len(str(datacor[i, 3])) == 3:
            Hr = 0
            Min = int(time[0])
            Sec = int(time[1:3])
        #如果時間的長度是4的話代表有分和秒
        elif len(str(datacor[i, 3])) == 4:
            Hr = 0
            Min = int(time[0:2])
            Sec = int(time[2:4])
        #如果時間的長度是5的話代表有小時、分和秒
        elif len(str(datacor[i, 3])) == 5:
            Hr = int(time[0])
            Min = int(time[1:3])
            Sec = int(time[3:5])
        print("time: " + time, "Hr: ", Hr, "Min: ", Min, "Sec: ", Sec)
        #判斷如果介在8:45~13:45的話，就留下來，否則就捨去
        if (Hr == 8 and Min >= 45) or (Hr >= 9 and Hr < 13) or \
           (Hr ==  13 and Min <= 45):
              print("符合條件!!")
              dataans[count] = datacor[i]
              count = count + 1
    print("三次過濾完畢!")
    return dataans, count
def conversion(dataans, count, dataname):
    data_k = pd.DataFrame(columns = ["成交日期","交割月份","成交時間","開盤價","最高價","最低價","收盤價","成交量"])
    freqk = 100
    countTX = count
    count = 0
    O = dataans[0, 4]
    H = dataans[0, 4]
    L = dataans[0, 4]
    C = dataans[0, 4]
    V = dataans[0, 5]
    print("第0筆資料", "開盤價:", O, " 最高價:", H, " 最低價:", L," 收盤價:", C, " 成交量:", V)
    data_k.loc[0, "成交時間"] = dataans[0, 3]
    for i in range(1, countTX):
        #如果此筆的交易時段跟上一筆交易時段同除以100後的商不同的話就代表此筆不是上一根K線的資料
        #是新的一根K線的資料
        if (dataans[i, 3] // freqk) != (dataans[i - 1, 3] // freqk):
            print("開盤價:", O, " 最高價:", H, " 最低價:", L," 收盤價:", C, " 成交量:", V)
            data_k.loc[count, "開盤價"] = O
            data_k.loc[count, "最高價"] = H
            data_k.loc[count, "最低價"] = L
            data_k.loc[count, "收盤價"] = C
            data_k.loc[count, "成交量"] = V
            count = count + 1
            data_k.loc[count, "成交時間"] = dataans[i, 3]
            O = dataans[i, 4]
            H = dataans[i, 4]
            L = dataans[i, 4]
            C = dataans[i, 4]
            V = dataans[i, 5]
        else:
            if dataans[i, 4] > H:
                H = dataans[i, 4]
            if dataans[i, 4] < L:
                L = dataans[i, 4]
            C = dataans[i, 4]
            V = V + dataans[i, 5]
            
    data_k.loc[count, "開盤價"] = O
    data_k.loc[count, "最高價"] = H
    data_k.loc[count, "最低價"] = L
    data_k.loc[count, "收盤價"] = C
    data_k.loc[count, "成交量"] = V
    data_k.loc[:, "交割月份"] = dataans[0, 2]
    data_k.loc[:, "成交日期"] = dataans[0, 0]
    data_k.to_csv(dataname+ '.csv',index = False, sep = ',',encoding='ANSI')
    return data_k
'''
df = read_csv('Daily_2018_12_22') 
dfcor, count = fillter_nonTX(df)
datacor, count = fillter_nonDeliverymonth(dfcor, count, '201901')
dataans, count = fillter_nonGeneraltransact(datacor, count)
ans = conversion(dataans, count, 'Daily_2018_12_22')
'''



