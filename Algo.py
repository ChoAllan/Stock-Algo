import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# grabs a specific stock you want 
df = web.DataReader('AMD', data_source='yahoo', start='2012-01-01', end ='2021-1-17')

df.shape

# print(df)

# stock chart of AMD 
plt.figure(figsize=(16, 8))
plt.title('Close History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


data = df.filter