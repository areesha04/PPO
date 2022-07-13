import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Dropout
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import r2_score
import os


def complie_data():
    pass


def train_test_split(timeseries, train_ratio):
    ''' Splits data into training and testing sets '''

    train_size = int(len(timeseries) * train_ratio)

    test_size = int(len(timeseries)*(1-train_ratio))

    train, test = timeseries[0:train_size, :], timeseries[train_size -
                                                          test_size:int(len(timeseries)-test_size), :]
    return train, test, (train_size, test_size)


def train_neural_network(X, y, testX, testY, look_back, epochs=800):

    model = Sequential()

    model.add(LSTM(25, input_shape=(1, look_back)))

    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X, y, epochs=epochs, batch_size=50,
                        validation_data=(testX, testY), verbose=1)

    return model, history.history


def save_plot(data, xlabel, ylabel, named):
    ''' Plots dataframe given as an input, label it and save it into plots folder'''

    plt.figure(figsize=(20, 8))

    plt.plot(data)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.savefig(f'./plots/{named}.png')


def save_history(history):

    plt.figure(figsize=(20, 8))

    plt.plot(history['loss'])

    plt.plot(history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./plots/loss.png')


def normalize(series):
    ''' Scales large values to real rumber between 0 and 1'''

    scaler = MinMaxScaler(feature_range=(0, 1))

    series = scaler.fit_transform(series)

    return scaler, series


def create_dataset(dataset, look_back=1):
    ''' Converts an array of values into a dataset matrix '''

    dataX, dataY = [], []

    for index in range(len(dataset)-look_back-1):

        lags = dataset[index:(index+look_back), 0]

        dataX.append(lags)

        dataY.append(dataset[index + look_back, 0])

    return np.array(dataX), np.array(dataY)


def pre_process(dataframe):
    ''' Preprocesses the given dataframe:

        1. Extracts dates for future use
        2. Save plot for initial data
        3. Extract target variable's timeseries data
        4. Normalize dataframe 

        '''

    dates = dataframe['Despatch Date']

    dataframe.set_index('Despatch Date', inplace=True)

    save_plot(dataframe, 'Despatch Date', 'Despatched Qty', 'Input data plot')

    despatched_qty = dataframe['Despatched Quantity'].values

    timeseries = despatched_qty.reshape(-1, 1)

    scaler, timeseries = normalize(timeseries)

    return dates, timeseries, scaler


def save_test_plot(timeseries, predictions, train_size, test_size):

    testPredictPlot = np.empty_like(timeseries)

    testPredictPlot[:, :] = np.nan

    testPredictPlot[train_size-test_size+look_back +
                    1:int(len(timeseries)-test_size)] = predictions

    plt.figure(figsize=(30, 8))

    plt.plot(scaler.inverse_transform(timeseries))

    plt.plot(testPredictPlot)

    plt.ylabel('Production')

    plt.xlabel('Days')

    plt.legend(['Original', 'Predicted'], loc='upper left')

    plt.savefig('./plots/performance.png')


def generate_forecast(timeseries, model, look_back, for_days=60):

    x_input = np.array([item[0] for item in timeseries[-look_back:]])

    input_window = list(x_input)

    forecast = []

    day = 0

    while(day < for_days):

        if (len(input_window) > look_back):

            x_input = np.array(input_window[1:])

            x_input = x_input.reshape((1, 1, look_back))

            yhat = model.predict(x_input, verbose=0)

            input_window.append(yhat[0][0])

            input_window = input_window[1:]

            forecast.append(yhat[0][0])

            day = day+1

        else:

            x_input = x_input.reshape((1, 1, look_back))

            yhat = model.predict(x_input, verbose=0)

            input_window.append(yhat[0][0])

            forecast.append(yhat[0][0])

            day = day+1

    return forecast


def evaluate_model(model, trainX, trainY, testX, testY, scaler):

    trainPredict = model.predict(trainX)

    trainPredict = scaler.inverse_transform(trainPredict)

    testPredict = model.predict(testX)

    testPredict = scaler.inverse_transform(testPredict)

    trainY = scaler.inverse_transform([trainY])

    testY = scaler.inverse_transform([testY])

    with open('model_performance.txt', 'a') as file:

        date_time = datetime.now()

        date_time = date_time.strftime("%d/%m/%Y %H:%M:%S")

        message = f'{[date_time]}\ntraining score: {r2_score(trainY[0], trainPredict[:,0])}\ntesting score: {r2_score(testY[0], testPredict[:,0])}'

        file.write(message)

        trainScore = math.sqrt(mean_squared_error(
            trainY[0], trainPredict[:, 0]))

        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

        rmse = '\nTrain Score: %.2f RMSE' % (
            trainScore) + '\n' + 'Test Score: %.2f RMSE\n\n' % (testScore)

        file.write(rmse)

        file.close()

        return testPredict


def save_forecast(dataframe, forecast):

    plt.figure(figsize=(30, 8))

    plt.plot(dataframe)

    plt.plot(forecast)

    plt.xlabel('Despatch Date')

    plt.ylabel('Despatched Quantity')

    plt.legend(['Old', 'Forecast'], loc='upper left')

    plt.savefig('./plots/forecast.png')

    forecast.to_excel('forecast.xlsx')
    forecast.to_json('forecast.json')


if __name__ == "__main__":

    if not os.path.exists("./plots"):
        os.mkdir("./plots")

    input_file_name = "./uploads/train_file.xlsx"

    np.random.seed(0)

    dataframe = pd.read_excel(input_file_name)

    dates, timeseries, scaler = pre_process(dataframe)

    training_data, testing_data, sizes = train_test_split(timeseries, 0.7)

    look_back = 50

    trainX, trainY = create_dataset(training_data, look_back)

    testX, testY = create_dataset(testing_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model, history = train_neural_network(
        trainX, trainY, testX, testY, look_back, epochs=800)

    save_history(history)

    test_predicted = evaluate_model(
        model, trainX, trainY, testX, testY, scaler)

    save_test_plot(timeseries, test_predicted, sizes[0], sizes[1])

    forecast = generate_forecast(timeseries, model, look_back, for_days=60)

    forecast = [[item] for item in forecast]

    forecast_df = [
        item for item in scaler.inverse_transform(np.array(forecast))]

    future_dates = [dates[-1:].iloc[-1] +
                    DateOffset(days=x)for x in range(0, 61)]

    forecast_frame = pd.DataFrame(
        forecast_df, index=future_dates[1:], columns=['Forecast'])

    save_forecast(dataframe, forecast_frame)