#import 存取Mariadb
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
print("1.讀取資料:")
#print("     Step 1 讀取資料: ")
new_data = pd.read_csv('D:/Python/專題/keras-LSTM(深度學習)/19980722_20150325.csv', engine = 'python')
#new_data = pd.read_csv('19980722_20150325.csv', engine = 'python')
new_data = new_data[857105:]
new_data.index = range(len(new_data))

new_data.set_index('Date', inplace = True)
new_data.index=pd.to_datetime(new_data.index)
#將分鐘數據聚合為日數據
data_open=new_data.loc[:,'Open'].resample('D').first().dropna()
data_high=new_data.loc[:,'High'].resample('D').max().dropna()
data_low=new_data.loc[:,'Low'].resample('D').min().dropna()
data_close=new_data.loc[:,'Close'].resample('D').last().dropna()
data_volume = new_data.loc[:, 'Volume'].resample('D').sum().dropna()
#將資料存到新的Dataframe中
new_data = pd.concat([data_open, data_high, data_low, data_close, data_volume], axis = 1)
#new_data.dropna(axis = 0, how = 'any')
#
new_data = new_data[new_data['Volume'] != 0]
new_data = new_data.reset_index()

#求出RSV
new_data['RSV'] = 100 * ((new_data['Close'] - new_data['Low'].rolling(window = 9).min()) /
  (new_data['High'].rolling(window = 9).max() - new_data['Low'].rolling(window = 9).min()))

new_data['RSV'].fillna(0, inplace = True)

#求出KD值
diction = {'K9' : [17], 'D9' : [39]}
for i in range(1, len(new_data.index)):
    K9_value = (1/3) * new_data['RSV'][i] + (2/3) * diction['K9'][i - 1]
    diction['K9'].append(K9_value)
    D9_value = (2/3) * diction['D9'][i - 1] + (1/3) * diction['K9'][i]
    diction['D9'].append(D9_value)

data_KD = pd.DataFrame(diction)
new_data['K9'] = data_KD['K9'].shift(8)
new_data['D9'] = data_KD['D9'].shift(8)

new_data['K9'].fillna(0, inplace = True)
new_data['D9'].fillna(0, inplace = True)

new_data['NextClose'] = new_data['Close'].shift(1)
new_data.fillna(method = 'bfill', inplace = True)


print("==================================================")
print("2.資料正規化:")
#print("     Step 1 定義正規化方式: ")
def normalize(df):
    newdf = df.copy()
    scaler = MinMaxScaler()
    #將資料裡的欄位壓縮成1 ~ 0之間的值，得以提升執行效率
    newdf['Open'] = scaler.fit_transform(df.Open.values.reshape(-1,1))
    newdf['High'] = scaler.fit_transform(df.High.values.reshape(-1,1))
    newdf['Low'] = scaler.fit_transform(df.Low.values.reshape(-1,1))
    newdf['Close'] = scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['Volume'] = scaler.fit_transform(df.Volume.values.reshape(-1, 1))
    newdf['NextClose'] = scaler.fit_transform(df.NextClose.values.reshape(-1, 1))
    return newdf
#print("     Step 2 進行正規化: ")
data_norm = normalize(new_data)
print(data_norm.head())

new_data = new_data.drop(['Volume'],1)
data_norm = data_norm.drop(['Volume'],1)
new_data = new_data.drop(['RSV'],1)
data_norm = data_norm.drop(['RSV'],1)
print("==================================================")
print("3.資料分割成訓練、測試:")
#df為傳進來的DataFrame,pastday為過去的天數，futureday為未來的天數
def data_helper(df, pastday, futureday):
    X_train, Y_train = [],[]
    #df.shape[0] - pastday - futureday的用意是為了不要超出範圍，因為你的array是三維度的
    #X_train的每個矩陣裡的資料皆是過去一天也就是取當天的資料，沒有影響
    #Y_train的每個矩陣裡的資料皆是取明天的資料，i會超出範圍、溢位，所以要扣掉futureday
    #而要扣掉pastday的原因是因為對X_train而言，當它取到最後一筆時會沒有下一筆資料可以取，所以要扣掉pastday
    for i in range(df.shape[0] - pastday - futureday):
        #從樣本中抽取出Open、High、Low、Volume、Close
        X_train.append(np.array(df.iloc[i : i + pastday,:-1]))
        #從樣本中抽取出NextClose
        Y_train.append(np.array(df.iloc[i : i + futureday,-1]))
        #Y_train.append(np.array(df.iloc[i + pastday : i + pastday + futureday,-1]))
        # == Y_train.append(np.array(df.iloc[i + pastday : i + pastday + futureday]["NextClose"])
        #X_train,Y_train皆是三維的矩陣
    return np.array(X_train), np.array(Y_train)

#分配、測試集(因為現在已過去資料當作樣本，無現在、未來資料，所以驗證集會從訓練集裡抽取出來，不會參與訓練
#            而測試集則是從樣本當中抽取出來)
def splitData(X, Y):
    #從X中抽取80%當作X_train
    X_train = X[:int(X.shape[0] * 0.8)]
    #從Y中抽取80%當作Y_train
    Y_train = Y[:int(Y.shape[0] * 0.8)]
    #從X中抽取20%當作X_test
    X_test = X[int(X.shape[0] * 0.8):]
    #從Y中抽取20%當作Y_test
    Y_test = Y[int(Y.shape[0] * 0.8):]
    
    #因為模型是一對一的模型，所以Y_train跟Y_test要二維的，
    #所以將Y_train、Y_test的三維透過reshape壓縮成二維
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])
    return X_train,Y_train,X_test,Y_test
    
print("4.建立模型:")
def build_model(shape):
    #可通過向Sequential模型傳遞第一個layer的list來建構該模型
    model = Sequential()
    model.add(LSTM(5, input_length = shape[1], input_dim = shape[2]))
    #model.add(Dropout(0.3))
    #model.add(Dropout(0.25))
    model.add(Dense(4, activation='relu'))
    #model.add(Dropout(0.15))
    model.add(Dense(3, activation='relu'))
    model.add(Dropout(0.15))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.15))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(2, activation='relu'))
    #model.add(Dropout(0.2))
    #此層為輸出層並用linear為激活函數
    model.add(Dense(1, activation='linear'))
    
    print(model.summary())
    model.compile(loss='mse',optimizer='adam')
    
    return model

X_train, Y_train = data_helper(data_norm.iloc[:,2:], 1, 1)
X_train, Y_train, X_test, Y_test = splitData(X_train, Y_train)

#建立模型
model = build_model(X_train.shape)
print("==================================================")
print("5.開始訓練模型:")
#使用Earlystopping來判斷當loss比上一個的loss還大時就經過parience個epoch就停止訓練
#callback = EarlyStopping(monitor = "loss", patience = 10, verbose = 1, mode = "auto")
#開始做訓練，把X_train跟Y_train丟進去模型做訓練，批次為1000次，每一次丟128的資料作訓練，驗證集是取訓練集的20%
train_history = model.fit(X_train, Y_train, batch_size = 7, 
                           epochs=5, validation_split = 0.2, verbose=1)
#儲存訓練好的模型
model.save('LSTM.h5')
#讀取模型
model_lstm = load_model('LSTM.h5')
'''
def backtest(dataframe, data_no, model_ls):
    deposits = []
    #資產為100萬
    dep = 1000000
    #淨賺
    benefit = 0
    deposits.append(dep)
    #持股的數量
    dos = 0
    #存放買那張期貨的收盤價
    reg = 0
    #停損點
    stlo = 0
    #停利點
    stbo = 0
    #勝時的賺錢次數    
    right_call = 0
    #勝時的賺錢金額
    right_score = 0
    #輸時的虧錢次數
    wrong_call = 0
    #輸時的賺錢金額
    wrong_score = 0
    #盈虧比
    pl = 0
    #df_backtest = data_no.iloc[:int(len(data_no.index) * 0.8),2:-1]
    df_backtest = data_no.iloc[:int(len(data_no.index) * 0.8),:-1]
    #df_backtest.index = range(len(df_backtest))
    X_backtest = np.array(df_backtest)
    #X_backtest = preprocessing.scale(X_backtest)
    X_backtest = X_backtest.reshape(X_backtest.shape[0],1 ,X_backtest.shape[1])
    #機器做預測
    forecast = model_ls.predict(X_backtest)
    #做多的買賣交易
    for i in range(1, len(df_backtest)):
        #未持股
        if dos == 0:
            #如果天數在三天以上時就做以下的判斷
            if i >= 2:
                #如果只有三天就判斷三天的KD值
                if i == 2:
                   if (df_backtest['K9'][i] and df_backtest['D9'][i]) < 20 and (df_backtest['K9'][i - 1] and df_backtest['D9'][i - 1]) < 20  and (df_backtest['K9'][i - 2] and df_backtest['D9'][i - 2]) < 20:
                            print("低檔鈍化，為超賣區，不做動作!!!")
                #如果K或D值 < 20 持續三天的話會產生低檔鈍化，不做任何動作
                elif i >= 3 and (df_backtest['K9'][i] and df_backtest['D9'][i]) < 20 and (df_backtest['K9'][i - 1] and df_backtest['D9'][i - 1]) < 20  and (df_backtest['K9'][i - 2] and df_backtest['D9'][i - 2]) < 20:
                        if (df_backtest['K9'][i - 3] and df_backtest['D9'][i - 3]) > 20:
                            print("低檔鈍化，為超賣區，不做動作!!!")
                #如果K或D值 < 20的話，明天就買進且判斷K大於D值且前一天的K值小於D值，代表有黃金交叉
                #符合黃金交叉後接著判斷明天的收盤價是否有大於今日的最高價，有的話就買進
                elif (df_backtest['K9'][i] and df_backtest['D9'][i]) < 20 and (df_backtest['K9'][i] >= df_backtest['D9'][i]) and (df_backtest['K9'][i - 1] < df_backtest['D9'][i - 1]) and \
                      (df_backtest['High'][i] < forecast[i]):
                    #買一張台指期
                    dos = 1
                    #記錄買的收盤價
                    reg = dataframe['Close'][i + 1]
                    #紀錄停損點，設定為買的那天的最低價 - 10%為停損點
                    stlo = dataframe['Low'][i + 1] - (dataframe['Low'][i + 1] * 0.2)
                    #紀錄停利點，設定為買的那天的最高價 + 10%為停利點
                    stbo = dataframe['High'][i + 1] + (dataframe['High'][i + 1] * 0.2)
            #如果KD值 < 20而且是黃金交叉的話就買進
            #符合黃金交叉後接著判斷今天的收盤價是否有大於昨日的最高價，有的話就買進
            elif (df_backtest['K9'][i] and df_backtest['D9'][i]) < 20 and (df_backtest['K9'][i] >= df_backtest['D9'][i]) and (df_backtest['K9'][i - 1] < df_backtest['D9'][i - 1]) and \
                 (df_backtest['High'][i] < forecast[i]):
                #買一張台指期
                dos = 1
                #記錄買的收盤價
                reg = dataframe['Close'][i + 1]
                #紀錄停損點，設定為買的那天的最低價 - 10%為停損點
                stlo = dataframe['Low'][i + 1] - (dataframe['Low'][i + 1] * 0.2)
                #紀錄停利點，設定為買的那天的最高價 + 10%為停利點
                stbo = dataframe['High'][i + 1] + (dataframe['High'][i + 1] * 0.2)
        #有持股
        else:
            if i >= 2:
                if i == 2:
                   if (df_backtest['K9'][i] and df_backtest['D9'][i]) > 80 and (df_backtest['K9'][i - 1] and df_backtest['D9'][i - 1]) > 80 and (df_backtest['K9'][i - 2] and df_backtest['D9'][i - 2]) > 80:
                        print("高檔鈍化，為超賣區，不做動作!!!")
                #連續三天K值或D值在80以上會產生高檔鈍化，不做任何動作
                elif i >= 3 and (df_backtest['K9'][i] and df_backtest['D9'][i]) > 80 and (df_backtest['K9'][i - 1] and df_backtest['D9'][i - 1]) > 80  and (df_backtest['K9'][i - 2] and df_backtest['D9'][i - 2]) > 80:
                    if (df_backtest['K9'][i - 3] and df_backtest['D9'][i - 3]) < 80:
                        print("高檔鈍化，為超賣區，不做動作!!!")
                #如果KD值 > 80的話，明天就賣出且判斷K等於D值且前一天的K值大於D值，代表有死亡交叉，明天就賣出
                #符合死亡交叉後接著判斷今天的收盤價是否有小於昨日的最低價，有的話就賣出
                elif (df_backtest['K9'][i] and df_backtest['D9'][i]) > 80 and (df_backtest['K9'][i] <= df_backtest['D9'][i]) and (df_backtest['K9'][i - 1] > df_backtest['D9'][i - 1]) and \
                     (df_backtest['Low'][i] < forecast[i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    stlo = 0
                    stbo = 0
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
                elif (stlo > dataframe['Close'][i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    #停損點歸0
                    stlo = 0
                    #停利點歸0
                    stbo = 0
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
                elif (stbo < dataframe['Close'][i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    #停損點歸0
                    stlo = 0
                    #停利點歸0
                    stbo = 0
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
            else:
                #如果K或D值 > 80的話，明天就賣出且判斷K等於D值且前一天的K值大於D值，代表有死亡交叉，明天就賣出
                #符合死亡交叉後接著判斷今天的收盤價是否有小於昨日的最低價，有的話就賣出
                if (df_backtest['K9'][i] and df_backtest['D9'][i]) > 80 and (df_backtest['K9'][i] <= df_backtest['D9'][i]) and (df_backtest['K9'][i - 1] > df_backtest['D9'][i - 1]) and \
                   (df_backtest['Low'][i] > forecast[i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
                elif (stlo > dataframe['Close'][i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    #停損點歸0
                    stlo = 0
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
                elif (stbo < dataframe['Close'][i]):
                    #期貨賣出歸0
                    dos = 0
                    #保證金 + (明日的收盤價 - 當初買的收盤價)並 * 200 
                    dep = dep + (dataframe['Close'][i + 1] - reg) * 200
                    #停利點歸0
                    stbo = 0
                    #判斷是否賣期貨時，是賺還是賠
                    if (dataframe['Close'][i + 1] - reg) > 0:
                        right_score = right_score + ((dataframe['Close'][i + 1] - reg) * 200)
                        right_call += 1
                    else:
                        wrong_score = wrong_score + ((reg - dataframe['Close'][i + 1]) * 200)
                        wrong_call += 1
        #把當前的保證金加進保證金序列中
        deposits.append(dep)
    #計算整體的報酬率    
    benefit = (float(dep - deposits[0]) / deposits[0]) * 100
    #計算總盈虧
    total_pl = (right_score) - (wrong_score)
    #計算盈虧比
    if wrong_score > 0:
        pl = (right_score / wrong_score) * 100
    else:
        pl = (right_score / 1) * 100
    #計算買賣的勝率            
    accuracy = (right_call / (right_call + wrong_call)) * 100
    #交易次數
    total_call = right_call + wrong_call
    df_backtest['deposit'] = pd.Series(deposits, index = df_backtest.index)
    ax1 = plt.subplot2grid((1,1),(0,0))
    df_backtest['deposit'].plot(ax = ax1, linewidth = 3, color = 'k')
    plt.show()
    print("----------------------------------------------------------")
    print("交易次數:", total_call)
    print("持", dos, "股!")
    print("整體的勝率: ", accuracy, "%")
    print("整體的報酬率: %.2f" % benefit , "%")
    print("剩餘的本金: ", dep, "元!")
    print("總盈虧:", total_pl)
    print("盈虧比:", pl)
    print("賺錢次數:", right_call)
    print("賺錢金額:", right_score)
    print("賠錢次數:", wrong_call)
    print("賠錢金額:", wrong_score)
    print("----------------------------------------------------------")

    #data_norm.to_csv(r'LSTM_KDPrediction.csv', sep = ',', encoding = 'ANSI',index = False)
    #print(deposits)
#backtest(new_data, data_norm, model_lstm)
'''
print("6.開始繪製圖形:")
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()
    
print("loss誤差執行結果:")
print(show_train_history(train_history,'loss','val_loss'))

#scores = model.evaluate(X_validation,Y_validation)
#使用evaluate計算出模型的準確度
scores = model_lstm.evaluate(X_test, Y_test)
#scores[1] = 1 - scores[0]
print('loss = ',scores)
#Y_actual = data_norm["NextClose"].values
#將訓練完的X_test跟Y_test做比較
Y_pred = model_lstm.predict(X_test)
plt.plot(Y_pred,color='red', label='LSTM Prediction')
plt.plot(Y_test,color='blue', label='Answer')
plt.legend()
plt.show()

#將資料存成csv檔
col = pd.DataFrame()
col = pd.DataFrame(Y_test, columns = ['Answer'])
col['LSTM_Prediction'] = pd.DataFrame(Y_pred)
col['LSTM_Loss'] = pd.DataFrame(train_history.history['loss'])
col.head()
#col.to_csv(r'LSTM_Prediction.csv', sep = ',', index = False)
'''





