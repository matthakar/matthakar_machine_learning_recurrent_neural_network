# matthakar_machine_learning_recurrent_neural_network

The purpose of this Python script is to forecast the mean temperature in London. To do this, it uses the machine learning library, PyTorch, and recurrent neural networks (RNNs), specifically LSTM models, on historical London weather data up to 2021. Although this script looks at weather data, this technology has various applications. Some examples include sales forecasting, market predictions, disease progression modeling, and drug design.

Given the right feature, hyperparameter, and parameter inputs, this model can forecast future features. We can validate the results by starting at a past date and forecasting features that we have actual data for. The resulting plot will help us see how well the actual data align with the forecasted results.

The data used for this script can be found here: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data

This script is split into 4 main parts after the data path is defined:

Part 1. Clean input data and engineer helpful features for the model

Part 2. Define hyperparameters and parameters

Part 3. Create RNN model function --> define features, normalization, model architecture, and plotting details

Part 4. Run the model and visualize the results

* Disclaimer --> I attached an example of the visualized model results with the default script definitions. Adjustments to the data, hyperparameters, parameters, feature inputs, normalization, and model architecture can all affect the output of this script and accuracy of the model.
