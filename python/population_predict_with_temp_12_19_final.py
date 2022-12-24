# -*- coding: utf-8 -*-
# 0. 呼叫GPU


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## 1. 取得資料
### 1.1 地點資料
#以下選定「港濱軸帶」試著設計預測模型，並且從自己的資料夾中取得分時num。其中這部分只需更改定點名稱(loc)即可。

#from google.colab import drive
#drive.mount('/content/gdrive')

# 引入模組
import pandas as pd

loc = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
#display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

### 1.2 取得天氣（溫度、降雨）資料

import pandas as pd
w_df = pd.read_csv("2022_Jun-Sep_溫度(C)_分景點.csv")
w_df = w_df[loc]
w_df = w_df.to_list()

r_df = pd.read_csv("/content/2022_Jun-Sep_降雨量(mm)_分景點_小雨1,大雨2,強雨3.csv",encoding= 'big5')
r_df = r_df[loc]
r_df = r_df.to_list()

### 1.3 取得票券資料

t_df = pd.read_csv("/content/赤崁樓_票券轉換7_8.csv")
t_df = t_df['tickets']
t_df = t_df.to_list()

### 1.4 取得假日資料

h_df = pd.read_csv("/content/2022holiday(hour).csv")
h_df = h_df.drop("Unnamed: 0", axis=1)

h_df['Sum']= h_df[['Weekend','New year','Spring festival','228 peace memorial day','Childrens’ day','Tomb sweeping day','May day','Dragon boat festival','Moon festival','National day','Highschool winter vacation','Highschool summer vacation','College winter vacation','College summer vacation','Huayuan night market opening day','long weekend','1st day of long weekend','Last day of long weekend','Last day of long weekend']].sum(1)

h_df

h_df = h_df['Sum']
h_df = h_df.to_list()

### 1.5 取得停車資料

p_df = pd.read_csv("/content/停車逐時資料_國華海安商圈.csv")
p_df = p_df['Parking']
p_df = p_df.to_list()


## 2. 建立預測模型
### 2.1 引入模型需要套件


import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout,GRU, Attention
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

### 2.2 資料處理（分割）
#更改部分：輸入時段input_hr。

input_hr = 48
input_dim = 6
output_hr = 5
last_hr = input_hr + output_hr

X=np.empty([len(num_arr) - last_hr, input_dim, input_hr], dtype = float)
y=np.empty([len(num_arr) - last_hr, output_hr], dtype = float)
for i in range(len(num_arr) - last_hr):
    #X[i][:][:] = [num_arr[i:i + input_hr], r_df[i:i + input_hr]] #, r_df[i:i + input_hr], w_df[i:i + input_hr], h_df[i:i + input_hr], t_df[i:i + input_hr], p_df[i:i + input_hr]
    X[i][:][:] = [num_arr[i:i + input_hr], r_df[i:i + input_hr], w_df[i:i + input_hr], h_df[i:i + input_hr], t_df[i:i + input_hr], p_df[i:i + input_hr]]#All variables
    #X[i][:][:] = [num_arr[i:i + input_hr], r_df[i:i + input_hr], w_df[i:i + input_hr], h_df[i:i + input_hr],  p_df[i:i + input_hr]]#No Ticket
    y[i][:] = num_arr[i + input_hr:i + last_hr]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.1)

### 2.3 模型建立

import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.layers import Layer
import keras.backend as K
from keras import Model,callbacks
from keras import optimizers
import torch

import math

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = callbacks.LearningRateScheduler(step_decay)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))

#### 赤崁

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

#pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、溫
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、票
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人票_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、假日
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人假日_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、停車
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人停車_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

"""#### 國華海安
票用赤崁
"""

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人雨_沒decay
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(LSTM(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(LSTM(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、溫
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)
pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、票
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人票_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、假日
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人假日_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、停車
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人停車_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

"""#### 孔廟
票用赤崁
"""

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)
pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)
pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、溫
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、票
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人票_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、假日
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人假日_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、停車
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人停車_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全＿decay
pred_model = Sequential()
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全_no decay
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_split = 0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)
pred_model.save('./{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全_no decay No Ticket
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)
pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/{}_人全NO票_200epoch_lr0.1＿best_model.h5'.format(loc))

"""#### 安平老街
票用古堡
"""

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
opt = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、溫
pred_model = Sequential()
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、票
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人票_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、假日
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人假日_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、停車
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人停車_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全 no decay
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
opt = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

"""#### 港濱軸帶"""

#12/14人、雨
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人雨_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、溫
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人溫_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、票
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人票_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、假日
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人假日_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、停車
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人停車_200epoch_lr0.1＿best_model.h5'.format(loc))

#12/14人、全
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 200
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)

loss_history = LossHistory()
lrate = callbacks.LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]
history = pred_model.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=32, 
   callbacks=callbacks_list, 
   verbose=2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.plot(train_loss, color='#FF4500')
plt.plot(val_loss, color='#326872')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2),
                Line2D([0], [0], color='#326872', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Training Loss', 'Validation Loss'])

plt.grid(True)

y_pred = pred_model(X_test)

pred_model.save('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/{}_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

"""### 2.4 模型實驗"""

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_GRU_with_attention(hidden_units, dense_units, input_shape, activation):
    x=Input(shape=input_shape)
    GRU_layer = GRU(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(GRU_layer)
    outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model=Model(x,outputs)
    model.compile(loss='mape', optimizer='adam')    
    return model

#GRU-Attention
pred_model = Sequential()

dense_units = 1024

pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.5))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.5))
pred_model.add(Dense(output_hr, activation = 'relu'))

x=Input(shape=(input_dim, 1024))
GRU_layer = GRU(dense_units, return_sequences=True, activation='relu')(x)
attention_layer = attention()(GRU_layer)
outputs=Dense(dense_units, trainable=True, activation='relu')(attention_layer)
pred_model=Model(x,outputs)


epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.summary()

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

#GRU-Attention
pred_model = Sequential()

dense_units = 1024

pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.5))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.5))
pred_model.add(Dense(output_hr, activation = 'relu'))

x=Input(shape=(input_dim, 1024))
GRU_layer = GRU(dense_units, return_sequences=True, activation='relu')(x)
attention_layer = attention()(GRU_layer)
outputs=Dense(dense_units, trainable=True, activation='relu')(attention_layer)
model=Model(x,outputs)


epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)

pred_model.summary()

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

#GRU
pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.5))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.5))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)


pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs= 1000 , batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.3))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.3))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)


pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs= 1000 , batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(GRU(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(GRU(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)


pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs= 1000 , batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(LSTM(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(LSTM(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

epochs = 1000
lr = 0.01
opt = optimizers.Adam(lr, decay=lr/epochs)


pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs= 1000 , batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(LSTM(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(LSTM(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.01)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 1000, batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model = Sequential()
pred_model.add(Dense(units = 1024, input_shape = (input_dim, input_hr), activation = 'relu'))
pred_model.add(LSTM(units = 1024, input_shape = (input_dim, 1024), return_sequences = True))
pred_model.add(Dropout(0.2))
pred_model.add(LSTM(units = input_hr, input_shape = (input_dim, 1024)))
pred_model.add(Dropout(0.2))
pred_model.add(Dense(output_hr, activation = 'relu'))

pred_model.summary()

optimizer = optimizers.Adam(0.1)

pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = optimizer)
history = pred_model.fit(X_train, y_train, epochs = 1000, batch_size = 32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)

pred_model(X_train)

y_train

"""### 2.5 導入最佳模型，視覺化

####預測資料

##### 國華海安
"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/國華海安商圈_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#326872",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#FF4500",scale = 0.5,plot_kws=dict(alpha=0.3))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Guohua Haian Shopping Area Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2, alpha = 0.3),
                Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 赤崁園區"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/赤崁園區_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '赤崁園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#326872",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#FF4500",scale = 0.5,plot_kws=dict(alpha=0.3))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Chikan Tower Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2, alpha = 0.3),
                Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 孔廟文化園區"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/孔廟文化園區_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '孔廟文化園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#326872",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#FF4500",scale = 0.5,plot_kws=dict(alpha=0.3))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Confucius Temple Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2, alpha = 0.3),
                Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 安平老街"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/安平老街_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '安平老街'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#326872",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#FF4500",scale = 0.5,plot_kws=dict(alpha=0.3))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Anping Old Street Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2, alpha = 0.3),
                Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 港濱軸帶"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/港濱軸帶_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#326872",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#FF4500",scale = 0.5,plot_kws=dict(alpha=0.3))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Yuguang Island Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#FF4500', lw=2, alpha = 0.3),
                Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])

"""####預測資料＿藍

##### 國華海安
"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/國華海安商圈/國華海安商圈_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.5,plot_kws=dict(alpha=2))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Guohua Haian Shopping Area Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 2),
                Line2D([0], [0], color='#99b6ac', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 赤崁園區"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/赤崁園區/赤崁園區_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '赤崁園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.5,plot_kws=dict(alpha=2))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Chikan Tower Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 2),
                Line2D([0], [0], color='#99b6ac', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 孔廟文化園區"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/孔廟文化園區/孔廟文化園區_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '孔廟文化園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.5,plot_kws=dict(alpha=2))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Confucius Temple Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 2),
                Line2D([0], [0], color='#99b6ac', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 安平老街"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/安平老街/安平老街_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '安平老街'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.5,plot_kws=dict(alpha=2))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Anping Old Street Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 2),
                Line2D([0], [0], color='#99b6ac', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])



"""##### 港濱軸帶"""

import tensorflow as tf

loaded_model = tf.keras.models.load_model('/content/gdrive/Shareddrives/智慧城市_ 5. 觀旅/data/LSTM Test/GRU model parameter/港濱軸帶/港濱軸帶_人全_200epoch_lr0.1＿best_model.h5'.format(loc))

loaded_model.summary()

y_pred_cmp = loaded_model(X_test)
y_pred_cmp[0]

# 引入模組
import pandas as pd
loc = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

min_df['date_hour']

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

for j in range(n):
  future_5hour = new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')
  new_df['date_hour'][len(min_df)+j]=future_5hour

new_df

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.5,plot_kws=dict(alpha=2))
plt.setp(g.collections, alpha=.3) #for the markers
plt.setp(g.lines, alpha=.3)       #for the lines


plt.title('Yuguang Island Predicted Crowd Numbers')
plt.xlabel('Date and Time')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 2),
                Line2D([0], [0], color='#99b6ac', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['After 5 hours', 'Before 48 hours'])

"""####歷史資料

##### 國華海安
"""

# 引入模組
import pandas as pd
loc = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

plt.title('Guohua Haian Shopping Area Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 赤崁園區"""

# 引入模組
import pandas as pd
loc = '赤崁園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
#display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
#display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

plt.title('Chikan Tower Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 孔廟文化園區"""

# 引入模組
import pandas as pd
loc = '孔廟文化園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

plt.title('Confucius Temple Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 安平老街"""

# 引入模組
import pandas as pd
loc = '安平老街'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

plt.title('Anping Old Street Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 港濱軸帶"""

# 引入模組
import pandas as pd
loc = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

plt.title('Yuguang Island Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""####歷史資料＿藍色

##### 國華海安
"""

# 引入模組
import pandas as pd
loc = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#294A6E",scale = 0.5)

plt.title('Guohua Haian Shopping Area Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#294A6E', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])

"""##### 赤崁園區"""

# 引入模組
import pandas as pd
loc = '赤崁園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#294A6E",scale = 0.5)

plt.title('Chikan Tower Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#294A6E', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 孔廟文化園區"""

# 引入模組
import pandas as pd
loc = '孔廟文化園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#294A6E",scale = 0.5)

plt.title('Confucius Temple Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#294A6E', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 安平老街"""

# 引入模組
import pandas as pd
loc = '安平老街'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#294A6E",scale = 0.5)

plt.title('Anping Old Street Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#294A6E', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])



"""##### 港濱軸帶"""

# 引入模組
import pandas as pd
loc = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
#display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()

min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
transform_dict = { 
                  "a":"date_hour"
                  }
min_df.rename(columns = transform_dict, inplace = True)
min_df.head()

from datetime import datetime, timedelta

initial = np.datetime64('2022-07-01 00:00')

for i in range(len(min_df)):
  every_hour = initial + np.timedelta64(i,'h')
  min_df['date_hour'][i]= every_hour

n = output_hr
new_index = pd.RangeIndex(n)
na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
new_df = pd.concat([min_df,na_df],axis=0)
new_df.reset_index(drop = True, inplace = True)
#display(new_df)
new_df_arr = new_df['num']
new_df_arr = new_df_arr.to_list()

for m in range(output_hr):
    new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
    new_num_arr = new_df['num']
    new_num_arr = new_num_arr.to_list()

min_df['date_hour']

h = 12
initial_date = '2022-07-01'
initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
min_df.loc[initial_date_filter]

initial_date

initial_date_filter

h = 12
end_date = '2022-08-31'
end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
min_df.loc[end_date_filter]

end_date_filter #中午12點

# initial_date = '2022-07-02'
# initial_date_filter = min_df[min_df['dt_date'] == initial_date][0:1].index[0]
# initial_date_filter = min_df['dt_date'][initial_date_filter]

# end_date = '2022-07-20'
# end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
# end_date_filter = min_df['dt_date'][end_date_filter]

import seaborn  as sns

px = 2/plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

sns.pointplot(x="date_hour", y="num", data=min_df[initial_date_filter:end_date_filter+1:24], color ="#294A6E",scale = 0.5)

plt.title('Yuguang Island Historical Crowd Numbers')
plt.xlabel('Date(day)')
plt.ylabel('Crowd Numbers')
plt.xticks(size=6, rotation=60)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#294A6E', lw=2)]

lines = plt.plot()
plt.legend(custom_lines, ['Historical Data'])

"""---
# 測試區
## 1. Underfitting test
"""

import joblib

err_list = []
for i in range(len(y_test)):
    err_list.append(float(abs(y_pred[i] - y_test[i])) / y_test[i] * 100)

print("avg: ", sum(err_list) / len(err_list), "(%), std: ", np.std(err_list), "(%)")

try:
    loaded_model = tf.keras.models.load_model('./{}_best_model.h5'.format(loc))

    y_pred_cmp = loaded_model(X_test)
    err_list_cmp = []
    for i in range(len(y_test)):
        err_list_cmp.append(float(abs(y_pred_cmp[0][i] - y_test[0][i])) / y_test[0][i] * 100)

    print("avg: ", sum(err_list_cmp) / len(err_list_cmp), "(%), std: ", np.std(err_list_cmp), "(%)")

    if(sum(err_list) / len(err_list) < sum(err_list_cmp) / len(err_list_cmp)):
        pred_model.save('./{}_best_model.h5'.format(loc))
        best_model = pred_model
    else:
        best_model = loaded_model
        err_list = err_list_cmp
except:
    pred_model.save('./{}_best_model.h5'.format(loc))
    best_model = pred_model

df_err = pd.DataFrame(columns = ['population', 'error'])
df_err['population'] = y_test
df_err['error'] = err_list
df_err.sort_values(by = ['population'], inplace = True)
df_err.reset_index(drop = True, inplace = True)

plt.plot(df_err['population'], df_err['error'])
plt.xlabel('population')
plt.ylabel('error(%)')
plt.grid(True)

pred_model(X_train)[0]

y_train[0]

X_train[0]

