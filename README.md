# Predicting Ethereum Price with Python and Machine Learning

This project aims to analyze and predict Ethereum prices using two different machine learning approaches: Support Vector Machine (SVM) with Scikit-learn and Long Short-Term Memory (LSTM) networks with TensorFlow and Keras.
By comparing these approaches, we can understand their effectiveness in handling time series data and their predictive capabilities for cryptocurrency prices.
This repository contains two ipynb files demonstrating how to analyze and predict the price of Ethereum using Python and various machine-learning techniques. The dataset used for this analysis spans from August 7, 2015, to June 2, 2024.

## LSTM with Keras and TensorFlow

The LSTM approach utilizes deep learning techniques to capture the temporal dependencies in the Ethereum price data. The key steps involved are:

1. **Data Preprocessing**
  * Load the Ethereum price data from a CSV file.
  * Normalize the features (Open, High, Low, Close, Volume) using MinMaxScaler.
2. **Sequence Creation**
  * Create sequences of a fixed length from the normalized data.
  * Split the sequences into training and testing sets.
3. **Model Building**
  * Construct an LSTM model with two LSTM layers, dropout layers, and a dense output layer.
  * Compile the model with the Adam optimizer and mean squared error loss.
4. **Model Training**
  * Train the LSTM model on the training data for a specified number of epochs.
5. **Model Evaluation**
  * Evaluate the trained model on the test data and calculate the test loss.

The LSTM model is implemented using Keras with TensorFlow as the backend.

## SVR with scikit-learn

The SVR approach employs a support vector machine algorithm for regression tasks. The key steps involved are:

1. **Data Preprocessing**
  * Load the Ethereum price data from a CSV file.
  * Normalize the features (Open, High, Low, Close, Volume) using MinMaxScaler.
2. **Feature Engineering**
  * Create additional features, such as moving averages or technical indicators, if desired.
3. **Model Building**
  * Split the data into training and testing sets.
  * Instantiate an SVR model with appropriate kernel and hyperparameters.
4. **Model Training**
  * Train the SVR model on the training data.
5. **Model Evaluation**
  * Evaluate the trained model on the test data and calculate performance metrics like mean squared error or R-squared.

The SVR model is implemented using the scikit-learn library.

## Conclusion
A Long Short-Term Memory (LSTM) model with Keras and TensorFlow is an appropriate choice for predicting Ethereum prices due to its ability to effectively model long-term dependencies and subtle patterns present in cryptocurrency data. The LSTM architecture, with its recurrent connections and gating mechanisms, can capture the intricate nuances and temporal relationships inherent in financial time series data.

LSTMs are particularly advantageous when dealing with long sequences of data, as they can selectively remember and forget information over extended periods, making them well-suited for tasks like cryptocurrency price prediction, where past price movements and market dynamics can influence future prices.

By leveraging the power of deep learning frameworks like Keras and TensorFlow, the LSTM model can be efficiently trained on large datasets, allowing it to learn complex patterns and make accurate predictions on unseen data.
While the performance of other models like SVR should also be evaluated, the LSTM approach with Keras and TensorFlow demonstrates promising results and aligns well with the characteristics of cryptocurrency price data, making it a strong candidate for this prediction task.

## References

- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Keras documentation](https://keras.io/)
- [Graphviz download](https://graphviz.gitlab.io/download/)
