# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import 日資料LSTM分析未加情緒分析 as dataLSTM
import numpy as np
import pandas as pd

print("1.讀取資料:")
#data是訓練的data，data_validation是驗證的data，data_test是測試的data
data = pd.read_csv('D:/Python/專題/keras-LSTM(深度學習)/201809-10.csv', engine = 'python')
data_validation = pd.read_csv('D:/Python/專題/keras-LSTM(深度學習)/201811.csv', engine = 'python')
data_test = pd.read_csv('D:/Python/專題/keras-LSTM(深度學習)/201812.csv', engine = 'python')
#                        names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume','mood'], header = 0)

#將不必要的欄位捨去
#data = data.drop(['contract'], 1)
data = data.drop(['date'], 1)
data = data.drop(['delivery-month'], 1)
data = data.drop(['Time'], 1)
data_validation = data_validation.drop(['date'], 1)
#data_validation = data_validation.drop(['contract'], 1)
data_validation = data_validation.drop(['delivery-month'], 1)
data_validation = data_validation.drop(['Time'], 1)
data_test = data_test.drop(['Date'], 1)
#data_test['Date'] = data_test['Date'].astype('str')
data_test = data_test.drop(['delivery-month'], 1)
data_test = data_test.drop(['Time'], 1)
#將NA的欄位用下一個欄位的值補上
print("==================================================")

def filter(df):
    C = df["Close"]
    df = df.drop(["Volume"], 1)
    df = df.drop(["Close"], 1)
    #df["Volume"] = V
    df["Close"] = C
    df["NextClose"] = C.shift(-1)   

    return df

data = dataLSTM.KD(data)
data_validation = dataLSTM.KD(data_validation)
data_test = dataLSTM.KD(data_test)
data = filter(data)
data_validation = filter(data_validation)
data_test = filter(data_test)
data.dropna(how = 'any',inplace = True)
data_validation.dropna(how = 'any', inplace = True)
data_test.dropna(how = 'any', inplace = True)
print("2.資料正規化:")
def normalize(df):
    newdf = df.copy()
    scaler = MinMaxScaler()
    #將資料裡的欄位壓縮成1 ~ 0之間的值，得以提升執行效率
    newdf['Open'] = scaler.fit_transform(df.Open.values.reshape(-1,1))
    newdf['High'] = scaler.fit_transform(df.High.values.reshape(-1,1))
    newdf['Low'] = scaler.fit_transform(df.Low.values.reshape(-1,1))
    newdf['Close'] = scaler.fit_transform(df.Close.values.reshape(-1,1))
    #newdf['Volume'] = scaler.fit_transform(df.Volume.values.reshape(-1, 1))
    newdf['NextClose'] = scaler.fit_transform(df.NextClose.values.reshape(-1, 1))
    newdf['K9'] = scaler.fit_transform(df.K9.values.reshape(-1, 1))
    newdf['D9'] = scaler.fit_transform(df.D9.values.reshape(-1, 1))
    return newdf

data = data.iloc[:-1,:]
data_validation = data_validation.iloc[:-1,:]
data_norm = normalize(data)
datavalidation_norm = normalize(data_validation)
datatest_norm = normalize(data_test)

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

X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data_helper(data_norm, datavalidation_norm, datatest_norm)
Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1])
Y_validation = Y_validation.reshape(Y_validation.shape[0], Y_validation.shape[1])
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1])

print("4.建立模型:")
def build_model(shape):
    model = Sequential()
    #6表示為輸出的神經元，input_length是時間步長也就是說
    #在三維度中每一個列裡面的列數是多少
    #input_dim是特徵表示在三維度中每一個列裡面的欄位數是多少
    model.add(LSTM(128, input_length = shape[1], input_dim = shape[2]))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.15))
    model.add(Dense(32, activation='relu'))
   # model.add(Dropout(0.15))
    model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.15))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    #輸出層要使用有效的adam的優化演算法，均方差誤差(mse)做為損失函數
    #為了避免必須採用手動方式來使得LSTM有輸出和復位其狀態，即使這在更新權重時
    #很容易做到，但是每輪訓練時還是把batch大小設置為樣本的大小
    model.compile(loss='mse',optimizer='adam')
    
    return model

model = build_model(X_train.shape)

print("==================================================")
print("5.開始訓練模型:")
#callback = EarlyStopping(monitor = "loss", patience = 10, verbose = 1, mode = "auto")
train_history = model.fit(X_train, Y_train, batch_size = len(X_train), 
                           epochs=100, validation_data = (X_validation, Y_validation), verbose=1)


model.save('LSTM.h5')
model_lstm = load_model('LSTM.h5')

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
scores = model.evaluate(X_validation, Y_validation)
print('loss = ',scores)

#model_lstm = load_model('LSTM.h5')
Y_pred = model.predict(X_test)
plt.plot(Y_pred,color='red', label='LSTM Prediction')
plt.plot(Y_test,color='blue', label='Answer')
plt.legend()
plt.show()

datatest_norm['LSTM_pred'] = Y_pred
pred = []
for i in range(0, len(datatest_norm)):
    if (datatest_norm.loc[i, 'LSTM_pred'] > datatest_norm.loc[i, 'Close']):
        pred.append("漲")
    elif (datatest_norm.loc[i,  'LSTM_pred'] == datatest_norm.loc[i, 'Close']):
        pred.append("NOT Action")
    else:
        pred.append("跌")
datatest_norm['Prediction'] = pred

test = []
for i in range(0, len(datatest_norm)):
    if (datatest_norm.loc[i, 'NextClose'] > datatest_norm.loc[i, 'Close']):
        test.append("漲")
    elif (datatest_norm.loc[i, 'NextClose'] == datatest_norm.loc[i, 'Close']):
        test.append("NOT Action")
    else:
        test.append("跌")
datatest_norm['Test'] = test

accuracytest = []
for i in range(0, len(datatest_norm)):
    if (datatest_norm.loc[i, 'Prediction'] == datatest_norm.loc[i, 'Test']):
        accuracytest.append("準確")
    else:
        accuracytest.append("錯誤")      
datatest_norm['accuracy'] = accuracytest

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
'''
#將資料存成csv檔
col = pd.DataFrame()
col = pd.DataFrame(Y_test, columns = ['Answer'])
col['LSTM_Prediction'] = pd.DataFrame(Y_pred)
col['LSTM_Loss'] = pd.DataFrame(train_history.history['loss'])
col.head()
col.to_csv(r'LSTM_FinalPrediction.csv', sep = ',', index = False)
'''

