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

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# from numba import jit, cuda
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def predict(place, output_hr):
    # place = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區
    # output_hr = 5
    print(type(output_hr))
    output_hr = int(0 if output_hr is None else output_hr)
    print(type(output_hr))

    min_df = pd.read_excel("D:/crowd-dashboard/data/台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
    min_df = min_df[min_df['attraction'] == place]
    min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
    min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
    min_df.reset_index(drop = True, inplace = True)
    #display(min_df)
    num_arr = min_df['num']
    num_arr = num_arr.to_list()

    ### 1.2 取得天氣（溫度、降雨）資料
    w_df = pd.read_csv("D:/crowd-dashboard/data/2022_Jun-Sep_溫度(C)_分景點.csv")
    w_df = w_df[place]
    w_df = w_df.to_list()

    r_df = pd.read_csv("D:/crowd-dashboard/data/2022_Jun-Sep_降雨量(mm)_分景點_小雨1,大雨2,強雨3.csv",encoding= 'big5')
    r_df = r_df[place]
    r_df = r_df.to_list()

    ### 1.3 取得票券資料

    t_df = pd.read_csv("D:/crowd-dashboard/data/票證/赤崁樓_票券轉換7_8.csv")
    t_df = t_df['tickets']
    t_df = t_df.to_list()

    ### 1.4 取得假日資料

    h_df = pd.read_csv("D:/crowd-dashboard/data/2022holiday(hour).csv")
    h_df = h_df.drop("Unnamed: 0", axis=1)
    h_df['Sum']= h_df[['Weekend','New year','Spring festival','228 peace memorial day','Childrens’ day','Tomb sweeping day','May day','Dragon boat festival','Moon festival','National day','Highschool winter vacation','Highschool summer vacation','College winter vacation','College summer vacation','Huayuan night market opening day','long weekend','1st day of long weekend','Last day of long weekend','Last day of long weekend']].sum(1)
    h_df = h_df['Sum']
    h_df = h_df.to_list()

    ### 1.5 取得停車資料

    p_df = pd.read_csv("D:/crowd-dashboard/data/停車/停車逐時資料_" + place + ".csv",encoding= 'utf-8')
    p_df = p_df['Parking']
    p_df = p_df.to_list()



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

    #if(place == "港濱軸帶" or place == "國華海安商圈"):
    if(place):
        """##### 港濱軸帶"""


        loaded_model = tf.keras.models.load_model("D:/crowd-dashboard/GRU model parameter/" + place + "/" + place + "_人全_200epoch_lr0.1＿best_model.h5".format(place))

        loaded_model.summary()

        y_pred_cmp = loaded_model(X_test)
        y_pred_cmp[0]

        # 引入模組
        #loc = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區

        min_df = pd.read_excel("D:/crowd-dashboard/data/台南_智發中心_成大產學合作_遊客分時.xlsx", sheet_name = "data sample")
        min_df = min_df[min_df['attraction'] == place]
        min_df['dt_date'] = pd.to_datetime(min_df['dt_date'])
        min_df.sort_values(by = ['dt_date', 'dt_hour'], inplace = True)
        min_df.reset_index(drop = True, inplace = True)

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
            every_hour = (initial + np.timedelta64(i,'h')).copy()
            min_df.loc[i, 'date_hour']= every_hour

        min_df['date_hour']

        n = output_hr
        new_index = pd.RangeIndex(n)
        na_df = pd.DataFrame(np.nan,index=new_index,columns=min_df.columns)
        new_df = pd.concat([min_df,na_df],axis=0)
        new_df.reset_index(drop = True, inplace = True)

        new_df_arr = new_df['num']
        new_df_arr = new_df_arr.to_list()

        # from uncertainties import ufloat
        # import pandas
        # import numpy

        # number_with_uncertainty = ufloat(2,1)

        # df = pandas.DataFrame({'a': [number_with_uncertainty]}) # This line works fine.

        # # create a new column with the correct dtype
        # df.loc[:, 'b'] = numpy.zeros(len(df), dtype=object)

        # df.loc[0,'b'] = ufloat(3,1) # This line now works.


        for m in range(output_hr):
            test = y_pred_cmp[0][m]
            #new_df.loc[:, 'num'] = np.zeros(len(new_df), dtype=object)
            #new_df.loc[len(min_df)+m, 'num'] = test
            new_df.loc[len(min_df)+m]['num'] = test
            #new_df['num'][len(min_df)+m] = test
            new_num_arr = new_df['num']
            new_num_arr = new_num_arr.to_list()

        print(new_df)
        min_df['date_hour']

        for j in range(n):
            future_5hour = (new_df['date_hour'] [len(min_df)+j-1]+ np.timedelta64(1,'h')).copy()
            new_df.loc[len(min_df)+j, 'date_hour']=future_5hour

        new_df

        import seaborn  as sns

        px = 2/plt.rcParams['figure.dpi']
        fig = plt.figure(figsize=(533.2*px, 125*px),constrained_layout =True, dpi = 400)

        ax = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(min_df)], color="#294A6E",scale = 0.5)
        g = sns.pointplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], color="#abd7a7",scale = 0.8)
        plt.setp(g.collections, alpha=.3) #for the markers
        plt.setp(g.lines, alpha=.3)       #for the lines

        bp = sns.barplot(x="date_hour", y="num", data=new_df[(len(min_df)-input_hr):len(new_df)], alpha=0)
            #data=min_df["num"]
        data = new_df[(len(min_df)-input_hr):len(new_df)]
        lst = data["num"].tolist()

        if (len(new_df[(len(min_df)-input_hr):len(new_df)]) < 20):
            labels = ax.bar_label(bp.containers[0])
        print(lst)

    else:
        pass
    #plt.title('Yuguang Island Predicted Crowd Numbers')
    plt.xlabel('Date and Time')
    plt.ylabel('Crowd Numbers')
    plt.xticks(size=6, rotation=60)


    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#abd7a7', lw=2, alpha = 1),
                    Line2D([0], [0], color='#99b6ac', lw=2)]

    lines = plt.plot()
    output_hr = str(output_hr)
    plt.legend(custom_lines, ['After' + output_hr + 'hours', 'Before 48 hours'])

    plt.savefig('D:/crowd-dashboard/預測資料new/'+ place +'.png', dpi='figure') 
    
    def Average(lst):
        return sum(lst) / len(lst)
  
    # Driver Code
    #lst = [15, 9, 55, 41, 35, 20, 62, 49]
    #average = Average(lst)
    average = np.nanmean(lst)
    print(average)
    #average = float(average)
    average = int(average)
    print(average)
    return average

place = '港濱軸帶'#赤崁園區,港濱軸帶,安平老街,國華海安商圈,孔廟文化園區
output_hr = 5
predict(place, output_hr)
