import OHLCV轉檔
import 下載台指期每日行情
import mysql.connector
import pandas as pd
def connect():
    conn = mysql.connector.connect(
            host = '127.0.0.1'
            ,user = 'root'
            ,passwd = 'nana60102'
            ,db = 'stocktx')
    
    cursor = conn.cursor()
    return cursor, conn

#create
def create(cursor, conn):
    sql = """CREATE TABLE TX (日期 CHAR(20) NOT NULL,
                              交割月份 CHAR(20) NOT NULL,
                              時間 CHAR(10) NOT NULL,
                              開盤價 CHAR(10) NOT NULL,
                              最高價 CHAR(10) NOT NULL,
                              最低價 CHAR(10) NOT NULL,
                              收盤價 CHAR(10) NOT NULL,
                              成交量 CHAR(10) NOT NULL)"""
    cursor.execute(sql)
    conn.commit()

def createacc(cursor, conn):
    sql = """CREATE TABLE accurate (日期 CHAR(20) NOT NULL,
                              時間 CHAR(20) NOT NULL,
                              今日收盤價 CHAR(10) NOT NULL,
                              預測明日收盤價 CHAR(10) NOT NULL)"""
    cursor.execute(sql)
    conn.commit()

#insert
def insert(cursor, conn, ans):
    #filename = 下載台指期每日行情.download()
    
    #filename = 'Daily_2018_12_24'
    
    #df = OHLCV轉檔.read_csv('D:/Python/專題/程式/' + filename)
                
    #dfcor, count = OHLCV轉檔.fillter_nonTX(df)
    #datacor, count = OHLCV轉檔.fillter_nonDeliverymonth(dfcor, count, '201901')
    #dataans, count = OHLCV轉檔.fillter_nonGeneraltransact(datacor, count)
    #ans = OHLCV轉檔.conversion(dataans, count, filename)
    
    #ans = pd.read_csv('D:/Python/專題/程式/Daily_2018_12_24.csv', sep = ',',engine='python')
    for i in range(0, len(ans)):
            #ans = pd.read_csv('D:/Python/專題/程式/Daily_2018_12_22.csv', sep = ',',engine='python')
            sql = "INSERT INTO tx (日期, 交割月份, 時間, 開盤價, 最高價,\
                                  最低價, 收盤價, 成交量) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
                        (ans.loc[i,"成交日期"], ans.loc[i,"交割月份"], ans.loc[i, "成交時間"], \
                        ans.loc[i, "開盤價"], ans.loc[i, "最高價"] , ans.loc[i, "最低價"], ans.loc[i, "收盤價"], \
                        ans.loc[i, "成交量"])
            try:
                  # Execute the SQL command
                  cursor.execute(sql)
                  # Commit your changes in the database
                  conn.commit()
            except:
                  # Rollback in case there is any error
                  print("error")
                  conn.rollback()
'''
def insertacc(cursor, conn):
    df = OHLCV轉檔.read_csv('D:/Python/專題/程式/LSTM_FinalPrediction.csv')
    for i in range(0, len(df)):
                #ans = pd.read_csv('D:/Python/專題/程式/Daily_2018_12_12.csv', sep = ',',engine='python')
            sql = "INSERT INTO accurate (日期, 今日收盤價, 預測明日收盤價) VALUES ('%s', '%s', '%s')" % \
                        (df.loc[i,"Answer"], df.loc[i,"LSTM"], df.loc[i, "成交時間"])
            try:
                  # Execute the SQL command
                  cursor.execute(sql)
                  # Commit your changes in the database
                  conn.commit()
            except:
                  # Rollback in case there is any error
                  print("error")
                  conn.rollback()

'''
#select
def select(cursor, conn):
    sql = "SELECT * FROM tx"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        #print(results)
        re = pd.read_sql_query(sql, conn)
        #print(re)
        #print(type(re))
    except:
        print ('unable to fetch data')
    conn.close()
    return re

#cursor, con = connect()
#create(cursor, con)
#insert(cursor, con)
#data = select(cursor, con)





