from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,LSTM,Conv1D,Flatten,MaxPooling1D,TimeDistributed
from keras.utils import np_utils 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import matplotlib.pyplot as plt
import 存取Mariadb
import numpy as np
import pandas as pd

def read_data():
    print("1.讀取資料:")
    data = pd.read_csv('19980722_20150325.csv', engine = 'python')
    data = data[857105:]
    data.index = range(len(data))
    data_validation = pd.read_csv('20150326_20150515.csv', engine = 'python')
    cursor, conn = 存取Mariadb.connect()
    data_test = 存取Mariadb.select(cursor, conn)
    return data, data_validation, data_test

#將不必要的欄位捨去
def filter_data(data, data_validation, data_test):
    data = data.drop(['Date'],1)
    data = data.drop(['Time'],1)
    data = data.drop(['Volume'],1)
    data_validation = data_validation.drop(['Date'],1)
    data_validation = data_validation.drop(['Time'],1)
    data_validation = data_validation.drop(['Volume'],1)
    
    data_test.columns = names = ['Date','交割月份','時間','Open', 'High', 'Low', 'Close', 'Volume']
    return data, data_validation, data_test

def transfor(data):
    data.set_index('Date', inplace = True)
    data.index = pd.to_datetime(data.index)
    data_Open=data.loc[:,'Open'].resample('D').first().dropna()
    data_High=data.loc[:,'High'].resample('D').max().dropna()
    data_Low=data.loc[:,'Low'].resample('D').min().dropna()
    data_Close=data.loc[:,'Close'].resample('D').last().dropna()
    new_data = pd.concat([data_Open, data_High, data_Low, data_Close], axis = 1)
    return new_data


def filter_close(df):
    C = df["Close"]
    df["NextClose"] = C.shift(-1)   
    df.dropna(how = 'any',inplace = True)
    return df

def KD(df):
    df['RSV'] = 100*((df['Close'].astype('float64') - df['Low'].rolling(window = 9).min()) /
      (df['High'].rolling(window = 9).max() - df['Low'].rolling(window = 9).min()))
    df['RSV'].fillna(0, inplace = True)
    
    diction = {'K9' : [17], 'D9' : [39]}
    for i in range(1, len(df.index)):
        K9_value = (1/3) * df['RSV'][i] + (2/3) * diction['K9'][i - 1]
        diction['K9'].append(K9_value)
        D9_value = (2/3) * diction['D9'][i - 1] + (1/3) * diction['K9'][i]
        diction['D9'].append(D9_value)
    
    data_KD = pd.DataFrame(diction, index = df.index)
    df = df.drop(['RSV'], 1)
    df['K9'] = data_KD['K9'].shift(8)
    df['D9'] = data_KD['D9'].shift(8)
    df['K9'].fillna(0, inplace = True)
    df['D9'].fillna(0, inplace = True)
    return df

#print("2.資料正規化:")
def normalize(df):
    newdf = df.copy()
    scaler = MinMaxScaler()
    #將資料裡的欄位壓縮成1 ~ 0之間的值，得以提升執行效率
    newdf['Open'] = scaler.fit_transform(df.Open.values.reshape(-1,1))
    newdf['High'] = scaler.fit_transform(df.High.values.reshape(-1,1))
    newdf['Low'] = scaler.fit_transform(df.Low.values.reshape(-1,1))
    newdf['Close'] = scaler.fit_transform(df.Close.values.reshape(-1,1))
    newdf['NextClose'] = scaler.fit_transform(df.NextClose.values.reshape(-1, 1))
    #newdf['K9'] = scaler.fit_transform(df.K9.values.reshape(-1, 1))
    #newdf['D9'] = scaler.fit_transform(df.D9.values.reshape(-1, 1))
    return newdf

#print("2.資料正規化:")
def normalizedata(df):
    newdf = df.copy()
    scaler = MinMaxScaler()
    #將資料裡的欄位壓縮成1 ~ 0之間的值，得以提升執行效率
    newdf['Open'] = scaler.fit_transform(df.Open.values.reshape(-1,1))
    newdf['High'] = scaler.fit_transform(df.High.values.reshape(-1,1))
    newdf['Low'] = scaler.fit_transform(df.Low.values.reshape(-1,1))
    newdf['Close'] = scaler.fit_transform(df.Close.values.reshape(-1,1))
    #newdf['Volume'] = scaler.fit_transform(df.Volume.values.reshape(-1, 1))
    newdf['K9'] = scaler.fit_transform(df.K9.values.reshape(-1, 1))
    newdf['D9'] = scaler.fit_transform(df.D9.values.reshape(-1, 1))
    return newdf

#print("3.資料分割成訓練、測試:")
def data_helper(df, df_val, df_test):
    X_train, Y_train = [],[]
    X_val, Y_val = [], []
    X_test, Y_test = [], []
    for i in range(df.shape[0]):
    #for i in range(df.shape[0] - pastday - futureday):
        X_train.append(np.array(df.iloc[i : i + 1,:-1]))
        Y_train.append(np.array(df.iloc[i : i + 1,[-1]]))
    for i in range(df_val.shape[0]):   
    #for i in range(df_val.shape[0] - pastday - futureday):
        X_val.append(np.array(df_val.iloc[i : i + 1,:-1]))
        Y_val.append(np.array(df_val.iloc[i : i + 1,[-1]]))
    for i in range(df_test.shape[0]):    
    #for i in range(df_test.shape[0] - pastday - futureday):
        X_test.append(np.array(df_test.iloc[i : i + 1,:-1]))
        Y_test.append(np.array(df_test.iloc[i : i + 1,[-1]]))  

    return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), np.array(X_test), np.array(Y_test)

#print("4.建立模型:")
def build_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_length = shape[1], input_dim = shape[2]))
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='linear'))
    #此層為輸出層並用linear為激活函數 
    model.compile(loss='mse',optimizer='adam')
    return model

def build_modelCNN(shape):
    model = Sequential()
    model.add(keras.layers.convolutional.Conv1D(filters=32, kernel_size=(1), strides=(1), padding='valid', input_shape=(1,6), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1)))
    model.add(keras.layers.convolutional.Conv1D(filters=64, kernel_size=(1), strides=(1), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1)))   
    model.add(keras.layers.convolutional.Conv1D(filters=96, kernel_size=(1), strides=(1), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1))) 
    model.add(keras.layers.convolutional.Conv1D(filters=128, kernel_size=(1), strides=(1), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1)))   
    model.add(keras.layers.convolutional.Conv1D(filters=160, kernel_size=(1), strides=(1), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1))) 
    model.add(keras.layers.convolutional.Conv1D(filters=192, kernel_size=(1), strides=(1), activation="relu"))
    model.add(keras.layers.MaxPool1D(pool_size=(1))) 
    model.add(Dropout(0.02))
    model.add(Flatten())
    model.add((Dense(1, activation='linear')))
    model.compile(loss='mse',optimizer='adam')
    print(model.summary())
    return model


def model_LSTM(X_train):
    model = build_model(X_train.shape)
    model.save('LSTM.h5')
    model_lstm = load_model('LSTM.h5')
    return model_lstm
    
def model_CNN(X_train):
    model = build_modelCNN(X_train.shape)
    model.save('CNN.h5')
    model_cnn = load_model('CNN.h5')
    return model_cnn

def model_choose(switch, a):
    switcher = {
        0: model_LSTM(a),
        1: model_CNN(a)
    }
    func =  switcher.get(switch, "nothing")
    return func

#print("6.開始繪製圖形:")
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()
    
def predict(datatest_norm):
    pred = []
    for i in range(0, len(datatest_norm)):
        if (datatest_norm.loc[i, 'LSTM_pred'] > datatest_norm.loc[i, 'Close']):
            pred.append("漲")
        elif (datatest_norm.loc[i,  'LSTM_pred'] == datatest_norm.loc[i, 'Close']):
            pred.append("NOT Action")
        else:
            pred.append("跌")
    datatest_norm['Prediction'] = pred
    return datatest_norm

def answ(datatest_norm):
    test = []
    for i in range(0, len(datatest_norm)):
        if (datatest_norm.loc[i, 'NextClose'] > datatest_norm.loc[i, 'Close']):
            test.append("漲")
        elif (datatest_norm.loc[i, 'NextClose'] == datatest_norm.loc[i, 'Close']):
            test.append("NOT Action")
        else:
            test.append("跌")
    datatest_norm['Test'] = test
    return datatest_norm

def accur(datatest_norm):
    accuracytest = []
    for i in range(0, len(datatest_norm)):
        if (datatest_norm.loc[i, 'Prediction'] == datatest_norm.loc[i, 'Test']):
            accuracytest.append("準確")
        else:
            accuracytest.append("錯誤")      
    datatest_norm['accuracy'] = accuracytest
    return datatest_norm

def accuracypredict(datatest_norm):
    ans = []
    accur = 0
    error = 0
    accuracy = 0
    for i in range(0, len(datatest_norm)):
        if (datatest_norm.loc[i, 'accuracy'] == "準確"):
            accur = accur + 1
        else:
            error = error + 1
    accuracy = accur / len(datatest_norm)
    print("預測正確的次數:", accur)
    print("預測失誤的次數:", error)
    print("準確率:", accuracy)
    datatest_norm.to_csv(r'LSTM_Prediction.csv', sep = ',', encoding = 'ANSI',index = False)
'''
#讀取檔案
data, data_validation, data_test = read_data()
#過濾檔案
data, data_validation, data_test = filter_data(data, data_validation, data_test)
#每筆行情轉成日資料
data_test = transfor(data_test)
#計算KD值
data = KD(data)
data_validation = KD(data_validation)
data_test = KD(data_test)

#Close放到最後一行當作輸出y，方便做訓練
data = filter_close(data)
data_validation = filter_close(data_validation)
data_test = filter_close(data_test)

#將最後一筆資料捨棄，因最後一筆的NextClose為Null
data = data.iloc[:-1,:]
data_validation = data_validation.iloc[:-1,:]
#資料正規化
data_norm = normalize(data)
datavalidation_norm = normalize(data_validation)
datatest_norm = normalize(data_test)

#分配train、validation、test
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data_helper(data_norm, datavalidation_norm, datatest_norm)
Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
Y_validation = Y_validation.reshape(Y_validation.shape[0], Y_validation.shape[1])
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

#建立model
model = model_choose(X_train)
print("5.開始訓練模型:")
train_history = model.fit(X_train, Y_train, batch_size = len(X_train), 
                          epochs=30, validation_data = (X_validation, Y_validation), verbose=1)

#畫出loss結果
print("loss誤差執行結果:")
print(show_train_history(train_history,'loss','val_loss'))
scores = model.evaluate(X_validation, Y_validation)
print('loss = ',scores)

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
datatest_norm = predict(datatest_norm)
#實際結果為漲、跌、平
datatest_norm = answ(datatest_norm)
#實際與預測的誤差結果
datatest_norm = accur(datatest_norm)
#統計、計算出準確率
accuracypredict(datatest_norm)
'''
 

