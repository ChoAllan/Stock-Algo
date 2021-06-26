import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.dataset_ops import DatasetSource
plt.style.use('fivethirtyeight')


# grabs a specific stock you want 
df = web.DataReader('AMD', data_source='yahoo', start='2012-01-01', end ='2021-6-25')

df.shape

# print(df)

# # stock chart of AMD 
# plt.figure(figsize=(16, 8))
# plt.title('Close History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()


# number of rows to train the model 
data = df.filter(['Close'])

dataset = data.values

training_data_len = math.ceil(len(dataset) * 0.8)

# print(training_data_len)

# scaling of data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# creation of training data set
train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60: i, 0])
    y_train.append(train_data[i, 0])
    # if i <= 61:
    #     print(x_train)
    #     print(y_train)


x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data
# print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(100))
model.add(Dense(1))

# 
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Testing Data Set
test_data = scaled_data[training_data_len - 60:, :]

# Creation of data sets for test

x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# models predicted price values

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions-y_test)**2)))
print(rmse)

train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions

#Visualize

# plt.figure(figsize=(16, 8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
# plt.show()

# print(valid)
amd_quote = web.DataReader('AMD', data_source='yahoo', start='2012-01-01', end ='2021-6-25')

new_df = amd_quote.filter(['Close'])

last_365_days = new_df[-365:].values

last_scale = scaler.transform(last_365_days)

X_test = []
X_test.append(last_scale)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)

print(pred_price)