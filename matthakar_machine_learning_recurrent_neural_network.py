import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

'''
The purpose of this Python script is to forecast the mean temperature in London. To do this, it uses the machine learning library, PyTorch, and recurrent neural networks (RNNs), specifically LSTM models, on historical London weather data up to 2021. Although this script looks at weather data, this technology has various applications. Some examples include sales forecasting, market predictions, disease progression modeling, and drug design.

Given the right feature, hyperparameter, and parameter inputs, this model can forecast future features. We can validate the results by starting at a past date and forecasting features that we have actual data for. The resulting plot will help us see how well the actual data align with the forecasted results.

The data used for this script can be found here: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data

This script is split into 4 main parts after the data path is defined:

Part 1. Clean input data and engineer helpful features for the model
Part 2. Define hyperparameters and parameters
Part 3. Create RNN model function --> define features, normalization, model architecture, and plotting details
Part 4. Run the model and visualize the results

* Disclaimer --> I attached an example of the visualized model results with the default script definitions. Adjustments to the data, hyperparameters, parameters, feature inputs, normalization, and model architecture can all affect the output of this script and accuracy of the model.
'''

# define input_data file path

input_data = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_machine_learning_recurrent_neural_network\london_weather.csv'

df = pd.read_csv(input_data)

'''
Part 1. Clean input data and engineer helpful features for the model
'''

# rename columns so they are more descriptive

df = df.rename(columns = {'date':'Date', 'cloud_cover':'Cloud Cover (oktas)', 'sunshine':'Sunshine (hrs)', 'global_radiation':'Global Radiation (W/m2)', 'max_temp':'Max Temp (°C)', 'mean_temp':'Mean Temp (°C)', 'min_temp': 'Min Temp (°C)', 'precipitation':'Precipitation (mm)', 'pressure':'Pressure (Pa)', 'snow_depth':'Snow Depth (cm)'})

# remove snow_depth column entirely since it has a lot of NAs --> dropping all columns with NA for this column could compromise the time series data by removing important dates, and this individual feature is not what I'm interested in

df = df.loc[:, df.columns != 'Snow Depth (cm)']

# remove rows from the df where at least one feature contains an NA value

df = df.dropna()

# convert Date column to datetime format

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# engineer new features based on the date column

df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# convert Date column format to include dashes

df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

'''
Part 2. Define hyperparameters and parameters
'''

# define hyperparameters

batch_size = 32 # batch_size x iterations = total samples in one epoch
n_past_training = 50 # number of past days we want to use to predict the future
n_future_training = 1 # number of days we want to look into the future based on the past days
num_epochs = 10 # total number of epochs
learning_rate = 0.01 # learning rate
n_past = 300 # number of days in the past where you want to start the forecast (can forecast things that already happened)
n_days_for_forecast = 1040 # number of days to forecast after the n_past day

# define parameters

visualization_start_date = '2016-01-01' # visualization_start_date 
date_column_name = 'Date' # define date_column_name
feature_to_forecast = 'Mean Temp (°C)' # define feature_to_forecast

# set the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Part 3. Define the RNN model function --> including features, normalization, model architecture, and plotting details
'''

def RNN_model(df, batch_size, n_past_training, n_future_training, num_epochs, learning_rate, n_past, n_days_for_forecast, visualization_start_date, date_column_name, feature_to_forecast):
    '''
    Prepare the input arrays
    '''
    # define multivariate feature columns so we can predict more than just one value in the future (don't include time)

    # print(df.columns)
    # feature_columns = ['Cloud Cover (oktas)', 'Sunshine (hrs)', 'Global Radiation (W/m2)', 'Max Temp (°C)', 'Mean Temp (°C)', 'Min Temp (°C)', 'Precipitation (mm)', 'Pressure (Pa)', 'Month', 'Quarter']
    feature_columns = ['Max Temp (°C)', 'Mean Temp (°C)', 'Min Temp (°C)', 'Month', 'Quarter'] # input features you want for a model
    input_df_for_training = df.loc[:, feature_columns]
    
    # print(input_df_for_training)

    # convert features (input_df_for_training) into a numpy array
    input_df_for_training = input_df_for_training.values

    # apply data normalization to the input df --> two options: StandardScaler() and MinMaxScaler()
    # scaler = StandardScaler() # better at dealing with outliers and normally distributed data, but does not constrain values to a specific range
    scaler = MinMaxScaler(feature_range=(0, 1)) # preserves the shape of the original distribution, but is sensitive to outliers

    scaler = scaler.fit(input_df_for_training)
    input_df_for_training = scaler.fit_transform(input_df_for_training)

    # print(input_df_for_training.shape)
    
    # create empty lists for training data, X_train is the input matrix, y_train is the output vector
    X_train = []
    y_train = []

    # reformat features (input_df_for_training) into the correct shape for an LSTM RNN: --> (n_samples (total in input_df_for_training) x timesteps (n_past_training) x n_features (input or output feature number))
    for i in range(n_past_training, len(input_df_for_training) - n_future_training + 1):
        X_train.append(input_df_for_training[i - n_past_training : i, 0 : input_df_for_training.shape[1]])
        y_train.append(input_df_for_training[i + n_future_training - 1 : i + n_future_training, 0])

    # convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_train)
    # print(y_train)

    '''
    Define the Autoencoder model (include multiple LSTM models for future predictions)
    '''

    # define model parameters
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]
    hidden_dim_1 = 64
    hidden_dim_2 = 32
    dropout_prob = 0.2

    # define the model architecture (including LSTM layers)
    class LSTM_model(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim_1, hidden_dim_2, dropout_prob):
            super(LSTM_model, self).__init__()
            self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
            self.dropout = nn.Dropout(dropout_prob)
            self.dense = nn.Linear(hidden_dim_2, output_dim)
        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x = self.dropout(x[:, -1, :])
            x = self.dense(x)
            return x

    # load the model
    model = LSTM_model(input_dim, output_dim, hidden_dim_1, hidden_dim_2, dropout_prob)

    # print(model)

    '''
    Convert numpy arrays to tensors and create dataloaders with the batch_size
    '''

    # convert numpy arrays to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # create dataLoader to define the batch_size
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    '''
    Run training loop
    '''
    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    verbose = True
    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # compute loss
            loss = criterion(outputs, labels)
            # backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print average loss after each epoch
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    '''
    Create list for forecast period dates
    '''

    # extract plotting dates from the original df
    train_dates = pd.to_datetime(df[date_column_name])

    forecast_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_forecast, freq='1d').tolist()

    '''
    Run forecast with trained model on future dates
    '''
    # set the model to evaluation mode for the test data --> this changes the behavior of the dropout and batch normalization layers
    model = model.eval()

    # makes forecast predictions using the model
    with torch.no_grad():
        forecast_tensor = model(X_train[-n_days_for_forecast:])

    # convert forecast_tensor to a numpy array
    forecast = forecast_tensor.numpy()

    # print(forecast)

    # ensures the forecast array has the same dimensions as the original input_df_for_training with x number of features
    forecast_copies = np.repeat(forecast, input_df_for_training.shape[1], axis=-1)

    # undo normalization
    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

    # print(y_pred_future)

    '''
    Define dates for plotting
    '''

    # convert timestamp to date
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    # print(forecast_dates)

    # convert the numpy arrays back into dfs for plotting
    forecast_df = pd.DataFrame({date_column_name:np.array(forecast_dates), feature_to_forecast:y_pred_future})
    forecast_df[date_column_name]=pd.to_datetime(forecast_df[date_column_name])

    original_df = df[[date_column_name, feature_to_forecast]]
    original_df[date_column_name] = pd.to_datetime(original_df[date_column_name])

    # cut off where the original_df starts for visualization purposes (want to see the forecast clearly by limiting the scale of the x-axis)
    original_df = original_df.loc[original_df[date_column_name] >= visualization_start_date]

    '''
    Plot the data
    '''
    
    # plot the data

    plt.figure(figsize=(9.33333, 4.666666))
    sns.set_style('whitegrid')
    sns.lineplot(x=date_column_name, y=feature_to_forecast, data=original_df, color='blue', label='Actual', linewidth = 1)
    sns.lineplot(x=date_column_name, y=feature_to_forecast, data=forecast_df, color='red', label='Forecast', linewidth=1)
    graph_name = 'Actual vs Forecasted ' + feature_to_forecast + ' for London' 
    plt.title(graph_name, fontsize=26, fontfamily='Calibri', fontweight='bold', pad=25)
    plt.xlabel(date_column_name, fontsize=16, fontfamily='Calibri', fontweight='bold', labelpad=15)
    plt.ylabel(feature_to_forecast, fontsize=16, fontfamily='Calibri', fontweight='bold', labelpad=10)
    plt.xticks(fontsize=14, fontfamily='Calibri', fontweight='bold', rotation=45)
    plt.yticks(fontsize=12, fontfamily='Calibri', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
Part 4. Run the model
'''

RNN_model(df, batch_size=batch_size, n_past_training=n_past_training, n_future_training=n_future_training, num_epochs=num_epochs, learning_rate=learning_rate, n_past=n_past, n_days_for_forecast=n_days_for_forecast, visualization_start_date=visualization_start_date, date_column_name=date_column_name, feature_to_forecast=feature_to_forecast)

