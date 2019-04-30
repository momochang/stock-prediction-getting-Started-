import OHLCV轉檔
import 下載台指期每日行情
import 存取Mariadb
import 日資料LSTM分析未加情緒分析 as dpmodel
import KD深度訓練 as kd
import matplotlib.pyplot as plt
import pandas as pd

def OHLCV(filename):
    print("讀取台指期今日行情的csv檔")
    df = OHLCV轉檔.read_csv(filename) 
    dfcor, count = OHLCV轉檔.fillter_nonTX(df)
    datacor, count = OHLCV轉檔.fillter_nonDeliverymonth(dfcor, count, '201901')
    dataans, count = OHLCV轉檔.fillter_nonGeneraltransact(datacor, count)
    ans = OHLCV轉檔.conversion(dataans, count, filename)
    return ans

def downloadTX():
    print("下載台指期今日行情資料")
    filename = 下載台指期每日行情.download()
    return filename

def Mariadb(ans):
    print("存取Mariadb")
    cursor, con = 存取Mariadb.connect()
    存取Mariadb.insert(cursor, con, ans)
    print("存取完畢!")
'''
def numbers_to_strings(a, filename):
    switcher = {
        0: filename = downloadTX(),
        1: ans = OHLCV(filename),
        2: Mariadb(ans),
        3:
        4: 
        #3: break
    }
    func =  switcher.get(a, "nothing")
    return func
'''
def 模型分析():
    #讀取檔案
    data, data_validation, data_test = dpmodel.read_data()
    #回傳原始資料用於回測
    return_data = data
    #過濾檔案
    data, data_validation, data_test = dpmodel.filter_data(data, data_validation, data_test)
    #每筆行情轉成日資料
    data_test = dpmodel.transfor(data_test)
    #計算KD值
    data = dpmodel.KD(data)
    data_validation = dpmodel.KD(data_validation)
    data_test = dpmodel.KD(data_test)
    
    #Close放到最後一行當作輸出y，方便做訓練
    data = dpmodel.filter_close(data)
    data_validation = dpmodel.filter_close(data_validation)
    data_test = dpmodel.filter_close(data_test)
    
    #將最後一筆資料捨棄，因最後一筆的NextClose為Null
    data = data.iloc[:-1,:]
    data_validation = data_validation.iloc[:-1,:]
    #資料正規化
    data_norm = dpmodel.normalize(data)
    datavalidation_norm = dpmodel.normalize(data_validation)
    datatest_norm = dpmodel.normalize(data_test)
    
    #分配train、validation、test
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = dpmodel.data_helper(data_norm, datavalidation_norm, datatest_norm)
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
    Y_validation = Y_validation.reshape(Y_validation.shape[0], Y_validation.shape[1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])
    
    #建立model
    switch = int(input("輸入0(LSTM)、1(CNN)："))
    model = dpmodel.model_choose(switch, X_train)
    print("5.開始訓練模型:")
    train_history = model.fit(X_train, Y_train, batch_size = len(X_train), 
                              epochs=100, validation_data = (X_validation, Y_validation), verbose=1)
    
    #畫出loss結果
    print("loss誤差執行結果:")
    print(dpmodel.show_train_history(train_history,'loss','val_loss'))
    scores = model.evaluate(X_validation, Y_validation)
    print('loss = ',scores)
    
    
    return datatest_norm, data_norm, return_data, X_test, Y_test, model

def 回測分析(data_origion, data_norm, model):
    '''
    data = pd.read_csv('D:/Python/專題/keras-LSTM(深度學習)/19980722_20150325.csv', engine = 'python')
    data = dpmodel.KD(data)
    data = data[857105:]
    data.index = range(len(data))
    data = data.drop(['Volume'],1)
    data_no = dpmodel.normalizedata(data)
    '''
    kd.backtest(data_origion, data_norm, model)

def 預測(datatest_norm, X_test, Y_test):
    #使用模型預測明天收盤價
    Y_pred = model.predict(X_test)
    #畫出預測、實際結果
    plt.plot(Y_pred,color='red', label='LSTM Prediction')
    plt.plot(Y_test,color='blue', label='Answer')
    plt.legend()
    plt.show()
    
    datatest_norm = datatest_norm.reset_index()
    datatest_norm['LSTM_pred'] = Y_pred
    #預測結果為漲、跌、平
    datatest_norm = dpmodel.predict(datatest_norm)
    #實際結果為漲、跌、平
    datatest_norm = dpmodel.answ(datatest_norm)
    #實際與預測的誤差結果
    datatest_norm = dpmodel.accur(datatest_norm)
    #統計、計算出準確率
    dpmodel.accuracypredict(datatest_norm)

txdown = int(input("是否需要下載台指期今日行情:是(1)、否(其他)："))
if (txdown == 1):
    filename = downloadTX()
    anohlc = int(input("是否需要轉成每筆交易資料:是(1)、否(其他)："))
    if (anohlc == 1):
        ans = OHLCV(filename)
        anmaria = int(input("是否需要存入資料庫:是(1)、否(其他)："))
        if (anmaria == 1):
            Mariadb(ans)
analy = int(input("是否需要深度學習進行分析:是(1)、否(其他)："))
if (analy == 1):
    datatest_norm, data_norm, data_origion, X_test, Y_test, model = 模型分析()
    anres = int(input("是否需要使用模型進行回測:是(1)、否(其他)："))
    if (anres == 1):
        回測分析(data_origion, data_norm, model)
    anpre = int(input("是否需要使用模型進行預測:是(1)、否(其他)："))
    if (anpre == 1):
        預測(datatest_norm, X_test, Y_test)




