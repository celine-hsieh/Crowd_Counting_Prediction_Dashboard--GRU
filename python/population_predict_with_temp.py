import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 引入模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout,GRU, Attention, Layer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import keras
from keras.layers import Input, Dense
import keras.backend as K
from keras import Model, optimizers

#1.1 地點資料
loc = '赤崁園區'

min_df = pd.read_excel("台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
min_df = min_df[min_df['attraction'] == loc]
min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
min_df.reset_index(drop = True, inplace = True)
#display(min_df)
num_arr = min_df['num']
num_arr = num_arr.to_list()


#1.2 取得天氣資料
import pandas as pd
w_df = pd.read_csv("2022_Jun-Sep_溫度(C)_分景點.csv")
w_df = w_df[loc]
w_df = w_df.to_list()

r_df = pd.read_csv("/content/2022_Jun-Sep_降雨量(mm)_分景點_小雨1,大雨2,強雨3.csv",encoding= 'big5')
r_df = r_df[loc]
r_df = r_df.to_list()

t_df = pd.read_csv("/content/赤崁樓_票券轉換7_8.xlsx - 7-8月份.csv")
t_df = t_df['tickets']
t_df = t_df.to_list()

#2. 建立預測模型
#2.2 資料處理（分割）更改部分：輸入時段input_hr
input_hr = 48
input_dim = 3
output_hr = 5
last_hr = input_hr + output_hr

X=np.empty([len(num_arr) - last_hr, input_dim, input_hr], dtype = float)
y=np.empty([len(num_arr) - last_hr, output_hr], dtype = float)
for i in range(len(num_arr) - last_hr):
    X[i][:][:] = [num_arr[i:i + input_hr], r_df[i:i + input_hr], t_df[i:i + input_hr]]#, r_df[i:i + input_hr]
    y[i][:] = num_arr[i + input_hr:i + last_hr]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.1)

#2.3 模型建立
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
model=Model(x,outputs)

pred_model.summary()

epochs = 1000
lr = 0.1
opt = optimizers.Adam(lr, decay=lr/epochs)


pred_model.compile(loss = 'MAPE', metrics=['accuracy'],optimizer = opt)
history = pred_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split = 0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch(s)')
plt.ylabel('Loss')
plt.grid(True)

y_pred = pred_model(X_test)