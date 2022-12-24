#呼叫GPU
import tensorflow as tf
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout, GRU, Attention
#from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.metrics import accuracy_score
import seaborn  as sns
from  matplotlib.ticker import FuncFormatter
# from numba import jit, cuda
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#歷史資料
# 國華海安
# 引入模組

def history(place, initial_date, end_date):
  #place = '國華海安商圈'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區
  print(place)
  str(initial_date)
  print(initial_date)
  str(end_date)
  print(end_date)
  min_df = pd.read_excel("D:/crowd-dashboard/data/台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
  min_df = min_df[min_df['attraction'] == place]
  #min_df = min_df[min_df['sub_category'] == '平日' or min_df['sub_category'] == "假日"]
  min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
  min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
  min_df.reset_index(drop = True, inplace = True)
  #display(min_df)
  num_arr = min_df['num']
  num_arr = num_arr.to_list()

  #print(num_arr)

  min_df = pd.concat([min_df,pd.DataFrame(columns=list('a'))])
  transform_dict = { 
                    "a":"date_hour"
                    }
  min_df.rename(columns = transform_dict, inplace = True)
  min_df.head()

  from datetime import datetime, timedelta
  print(1)
  #initial = np.datetime64('2022-07-01 00:00')
  initial = np.datetime64('2022-07-01 00:00')
  #pd.set_option('mode.chained_assignment', None)

  for i in range(len(min_df)):
    every_hour = (initial + np.timedelta64(i,'h')).copy()
    min_df.loc[i, 'date_hour'] = every_hour

  # print(2)
  # output_hr = 5
  # n = output_hr
  # #n = 5
  # new_index = pd.RangeIndex(n)
  # na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
  # new_df = pd.concat([min_df,na_df],axis=0)
  # new_df.reset_index(drop = True, inplace = True)
  # #display(new_df)
  # new_df_arr = new_df['num']
  # new_df_arr = new_df_arr.to_list()

  # for m in range(output_hr):
  #     new_df['num'][len(min_df)+m] = y_pred_cmp[0][m]
  #     new_num_arr = new_df['num']
  #     new_num_arr = new_num_arr.to_list()

  #min_df['date_hour']
  print(3)
  h = 12
  #print(type(initial_date))
  #initial_date = '2022-07-01'
  initial_date = str(initial_date)
  print(type(initial_date))
  initial_date_filter = min_df[min_df['dt_date'] == initial_date][h:h+1].index[0] #中午12點
  #min_df.loc[initial_date_filter]


  h = 12
  #print(end_date)
  #end_date = '2022-08-31'
  end_date = str(end_date)
  print(type(end_date))
  end_date_filter = min_df[min_df['dt_date'] == end_date][h:h+1].index[0] #中午12點
  #min_df.loc[end_date_filter]


  # initial_date = '2022-07-02'
  #initial_date_filter = min_df[min_df['sub_category'] == '平日'initial_date][0:1].index[0]
  # initial_date_filter = min_df['dt_date'][initial_date_filter]

  # end_date = '2022-07-20'
  # end_date_filter = min_df[min_df['dt_date'] == end_date][0:1].index[0]
  # end_date_filter = min_df['dt_date'][end_date_filter]
  print(min_df[initial_date_filter:24])
  #import seaborn  as sns
  print(4)
  px = 2/plt.rcParams['figure.dpi']
  fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)
  poeple = "num"
  ax = sns.pointplot(x="date_hour", y= poeple, data=min_df[initial_date_filter:end_date_filter+1:24], color ="#c48368",scale = 0.5)

  min_df["num"] = min_df["num"].astype(int)
  bp = sns.barplot(x="date_hour", y=poeple, data=min_df[initial_date_filter:end_date_filter+1:24], alpha=0)
  #data=min_df["num"]
  data = min_df[initial_date_filter:end_date_filter+1:24]
  lst = data["num"].tolist()
# use the containers of the barplot to generate the labels
  if (len(min_df[initial_date_filter:end_date_filter+1:24]) < 20):
    labels = ax.bar_label(bp.containers[0])
  print(lst)
  
  #plt.title(place + ' 歷史人流')
  plt.xlabel('Date(day)')
  plt.ylabel('Crowd Numbers')
  plt.xticks(size=6, rotation=60)
  #plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

  from matplotlib.lines import Line2D
  custom_lines = [Line2D([0], [0], color='#c48368', lw=2)]

  lines = plt.plot()
  plt.legend(custom_lines, ['Historical Data'])
  plt.savefig('D:/crowd-dashboard/歷史資料new/'+ place +'.png', dpi='figure') 
  def Average(lst):
    return sum(lst) / len(lst)
  
  # Driver Code
  #lst = [15, 9, 55, 41, 35, 20, 62, 49]
  average = Average(lst)
  average = int(average)
  print(average)
  return average

# place = place = '孔廟文化園區'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區
# initial_date = '2022-07-05'
# end_date = '2022-07-20'
# history(place, initial_date, end_date)