# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("1.讀取資料:")
#print("     Step 1 讀取資料: ")
data = pd.read_csv('D:/Python/期貨每日交易行情/201808.csv', engine = 'python')
print(data.head())

data = data.drop(['contract'], 1)
data = data.drop(['delivery-month'], 1)
data = data.drop(['time'], 1)
data.dropna(how = 'any',inplace = True)
print(data.tail())

labelsvol = []
labelsclo = []
labelsmood = []
for i in range(0, len(data.index)):
    labelsvol.append(data['volume'][i])
    labelsclo.append(data['close'][i])
    labelsmood.append(data['mood'][i])
        #print(i)
data = data.drop(['volume'], 1)
data = data.drop(['close'], 1)
data = data.drop(['mood'], 1)
data['volume'] = pd.Series(labelsvol, index = data.index)
data['mood'] = pd.Series(labelsmood, index = data.index)
data['close'] = pd.Series(labelsclo, index = data.index)
#dealdata(data, pd.Series(key))
print(data.head())
print("==================================================")
print("2.資料正規化:")
#print("     Step 1 定義正規化方式: ")


def normalize(df):
    newdf = df.copy()
    scaler = MinMaxScaler()
    #scaler.fit(newdf)
    newdf['open'] = scaler.fit_transform(df.open.values.reshape(-1,1))
    newdf['high'] = scaler.fit_transform(df.high.values.reshape(-1,1))
    newdf['low'] = scaler.fit_transform(df.low.values.reshape(-1,1))
    #newdf['Min'] = scaler.fit_transform(df.Min.values.reshape(-1,1))
    newdf['close'] = scaler.fit_transform(df.close.values.reshape(-1,1))
    newdf['volume'] = scaler.fit_transform(df.volume.values.reshape(-1, 1))
    #newdf['Stock_trade'] = scaler.fit_transform(df.Stock_trade.values.reshape(-1,1))
    return newdf
#print("     Step 2 進行正規化: ")
data_norm = normalize(data)
print(data_norm.head())


print("==================================================")
print("3.資料分割成訓練、測試:")
def data_helper(df, time_frame):
    #print("     Step 1 算資料的維度: ")
    number_features = len(df.columns)
    #print("     Step 2 將資料(dataframe)轉換成np.array: ")
    datavalue = df.as_matrix()
    #print(datavalue)
    #print(type(datavalue))
    
    result = []
    #print("     Step 3從 datavalue 的第0個跑到倒數第 time_frame+1 個")
    for index in range(len(datavalue) - (time_frame + 1)): 
        # 逐筆取出 time_frame + 1 個K棒數值做為一筆 instance
        result.append(datavalue[index: index + (time_frame + 1)]) 
    
    result = np.array(result)
    # print(type(result))
    # 取 result 的前80% instance做為訓練資料，shape[0]為取列的數量
    number_train = round(0.8 * result.shape[0])
    
    # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
    x_train = result[:int(number_train), :-1] 
    # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
    y_train = result[:int(number_train), -1][:,-1] 
    
    # 測試資料
    x_test = result[int(number_train):, :-1]
    y_test = result[int(number_train):, -1][:,-1]
    
    # 將資料組成變好看一點
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], number_features).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], number_features).astype('float32')
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    return [x_train, y_train, x_test, y_test]
# 以20天為一區間進行股價預測

X_train, y_train, X_test, y_test = data_helper(data_norm, 20)

#data_helper(data_norm, 20)

print("==================================================")
print("4.建立模型:")
def build_model(input_length, input_dim):
    #d = 0.3
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(16,activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(1,activation='linear'))
    print(model.summary())
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    
    return model
# 20天、5維
model = build_model( 20, 6 )

print("==================================================")
print("5.開始訓練模型:")
train_history = model.fit(X_train, y_train, batch_size=200, 
                           epochs=50, validation_split=0.2, verbose=1)
print("==================================================")
print("6.開始繪製圖形:")
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train_history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'], loc='upper left')
    plt.show()
#print("    Step 4 畫出accuracy執行結果: ")
#print("accuracy執行結果:")
#print(show_train_history(train_history,'acc','val_acc'))    
print("loss誤差執行結果:")
print(show_train_history(train_history,'loss','val_loss'))

scores = model.evaluate(X_test,y_test)
scores[1] = 1 - scores[0]
print('accuracy = ',scores[1])

def denormalize(df, norm_value):
    original_value = df['close'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    return denorm_value
# 用訓練好的 LSTM 模型對測試資料集進行預測
pred = model.predict(X_test)
# 將預測值與正確答案還原回原來的區間值
denorm_pred = denormalize(data, pred)
denorm_ytest = denormalize(data, y_test)

plt.plot(denorm_pred,color='red', label='Prediction')
plt.plot(denorm_ytest,color='blue', label='Answer')
plt.legend(loc='best')
plt.show()














