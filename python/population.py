import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
# 引入模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout
#from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
# from numba import jit, cuda
# import numba as nb
# import tensorflow as tf
# import os
#os.environ["CUDA_VISIBLE_DEVICES"]='0' 
#@jit(target_backend='cuda') 
#@cuda.jit
#@nb.vectorize(target='cuda')
#os.environ[ 'LOG LEVEL'] = '2'

#!/home/ds/.conda/envs/billy/bin/python
is_cuda = torch.cuda.is_available()
#torch.cuda.device_count()
#torch.cuda.current_device()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

def map_harvest(place):
    #1. 取得資料
    min_df = pd.read_excel("D:/crowd-dashboard/data/台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
    # if(seleted==0):
    #     place = "國華海安商圈"
    # elif (seleted==1):
    #     place = '孔廟文化園區'
    # elif (seleted==2):
    #     place = '安平老街'
    # elif (seleted==3):
    #     place = '港濱軸帶'
    # elif (seleted==4):
    #     place = '赤崁園區'
    min_df = min_df[min_df['attraction'] == place]
    min_df.reset_index(drop = True, inplace = True)
    #display(min_df)
    num_arr = min_df['num']


    #2.2 資料處理（分割）
    input_hr = 48

    X=np.empty([len(num_arr) - input_hr, input_hr], dtype = int)
    y=np.empty(len(num_arr) - input_hr, dtype = int)
    for i in range(len(num_arr) - input_hr):
        X[:][i] = num_arr[i:i + input_hr]
        y[i] = num_arr[i + input_hr]

    X_train, X_valid, y_valid, y_valid = train_test_split(X, y, random_state = 0, test_size = 0.2)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, X_valid.shape[1]))

    

    #2.3 模型建立
    pred_model = Sequential()
    pred_model.add(LSTM(units = input_hr, input_shape = (1, input_hr), activation = 'relu'))
    pred_model.add(Dense(1, activation = 'relu'))
    pred_model.compile(loss = 'MAPE', optimizer = 'adam', metrics = ['accuracy'])
    history = pred_model.fit(X_valid, y_valid, validation_split =0.1, epochs = 150, batch_size = 32, verbose = 1)


    #plt.rcParams["figure.figsize"] = [200,8]
    # f = plt.figure()
    # plt.figure().set_figwidth(200)
    # plt.figure().set_figheight(8)
    #plt.figure(figsize=(3.841, 7.195), dpi=100)

    px = 2/plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(533.2*px, 125*px), constrained_layout=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.xlabel('Epoch(s)')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)

    plt.savefig('D:/crowd-dashboardEN/python/'+ place +'.png', dpi='figure') 
    #plt.savefig('D:/crowd-dashboardEN/python/test2.jpg') 
    #plt.rcParams["figure.figsize"] = (200,8)
    #return pred_model
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#map_harvest(2)