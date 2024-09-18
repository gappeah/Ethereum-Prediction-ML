# Predicting Ethereum Price with Python and Machine Learning

This project predicts the future price of Ethereum using historical data and a machine learning approach. It leverages LSTM (Long Short-Term Memory) neural networks, a type of recurrent neural network well-suited for time-series forecasting, to model and predict Ethereum prices.

## Project Overview

The primary objective of this project is to predict Ethereum prices using historical data. The model is trained using an LSTM network and the data consists of Ethereum prices from August 2015 to September 2024. The predictions are evaluated using metrics such as **Mean Squared Error (MSE)**, **R-squared (R²)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)** and **Accuracy** (though accuracy is unconventional in regression tasks, it is also computed here).

### Key Features

- Utilises LSTM for time-series forecasting.
- Preprocesses Ethereum price data and splits it into training and testing sets.
- Evaluates the model based on error metrics such as MSE and R².
- Visualises actual vs. predicted Ethereum prices.

## What is Ethereum 
Ethereum is a decentralised, global platform that uses blockchain technology to enable the creation of digital applications and currencies. It's known for its native cryptocurrency, ether (ETH), and is often mentioned alongside Bitcoin as a leader in the cryptocurrency and blockchain space. 
Ethereum is a decentralized, open-source blockchain platform that enables developers to build and deploy decentralized applications (DApps) through the use of smart contracts. It was proposed by Vitalik Buterin in 2013 and officially launched on July 30, 2015. The native cryptocurrency of the Ethereum network is Ether (ETH), which serves multiple purposes, including transaction fees and as a means of value transfer.

### Blockchain Architecture
Ethereum operates on a blockchain, which is a distributed ledger technology that records transactions in a secure and immutable manner. Each block in the Ethereum blockchain contains a cryptographic hash of the previous block, creating a chain that ensures data integrity. This architecture allows for a decentralized network where no single entity has control over the entire blockchain.

### Smart Contracts
Smart contracts are self-executing contracts with the terms directly written into code. They run on the Ethereum Virtual Machine (EVM), which allows any code compatible with EVM to execute on the Ethereum network. Smart contracts facilitate, verify, or enforce the negotiation or performance of a contract without intermediaries, thus reducing costs and increasing efficiency[1][2].

### Ethereum Virtual Machine (EVM)
The EVM is a runtime environment for executing smart contracts and DApps on the Ethereum blockchain. It abstracts the underlying complexity of the network, allowing developers to write code in various programming languages that can be compiled into EVM bytecode. The EVM operates in a deterministic manner, ensuring that all nodes reach consensus on the state of the blockchain after executing transactions[1][4].

### Consensus Mechanism
Ethereum transitioned from a Proof-of-Work (PoW) to a Proof-of-Stake (PoS) consensus mechanism with an upgrade known as "the Merge" on September 15, 2022. In PoS, validators are chosen to create new blocks based on the amount of ETH they hold and are willing to "stake" as collateral. This shift significantly reduced Ethereum's energy consumption by approximately 99% compared to its previous mining model[2][5].

### Gas Fees
Transactions on the Ethereum network require gas fees, which are paid in ETH. Gas serves as a measure of computational effort required to execute operations like transactions and smart contract executions. The fee structure incentivizes miners/validators to process transactions and prevents spam attacks on the network by requiring users to pay for resources consumed.

### Use Cases
Ethereum supports various applications across multiple domains:
- **Decentralized Finance (DeFi)**: Platforms built on Ethereum allow users to lend, borrow, trade, and earn interest without traditional financial intermediaries.
- **Non-Fungible Tokens (NFTs)**: Ethereum provides a framework for creating unique digital assets that represent ownership of specific items or content.
- **Decentralized Autonomous Organizations (DAOs)**: These entities operate through smart contracts, allowing members to govern collectively without centralized control.

## Dataset
The dataset used in this project contains Ethereum prices from 2015 to 2024. It is stored in a CSV file (`ethereum_2015-08-07_2024-09-08.csv`), which includes date-wise prices and other relevant features. The data is preprocessed, normalised, and split into training and test sets before being fed into the LSTM model.

## Setup Instructions

### Prerequisites

Ensure you have Python installed (preferably version 3.7 or higher). You'll also need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `keras`
- `pydotplus`
- `graphviz`
- `pydot`
- `kerastuner`

To install the required packages, run the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras kerastuner pydot pydotplus graphviz
```

### Files

- `main_lstm.ipynb`: Jupyter notebook containing the main code for data preprocessing, model training, and evaluation.
- `ethereum_2015-08-07_2024-09-08.csv`: Historical Ethereum price data used to train and test the model.


## Running the Project

1. **Clone the repository** or download the necessary files.
   ```bash
   git clone https://github.com/gappeah/https://github.com/gappeah/Ethereum-Prediction-ML
   ```

2. **Prepare the dataset**: Ensure the `ethereum_2015-08-07_2024-09-08.csv` file is in the working directory.

3. **Run the Jupyter notebook**: Open the `main_lstm.ipynb` file in Jupyter Notebook. The notebook is divided into several steps:
    - Data loading and preprocessing.
    - Splitting the data into training and testing sets.
    - LSTM model creation and training.
    - Model evaluation and visualisation of results.

    You can run the notebook cell-by-cell to execute the entire workflow.

### Code Example

Here's an example of how the LSTM model is defined and trained:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64)
```

Once trained, you can predict Ethereum prices and evaluate the model using the following metrics:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Flatten predictions
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

### Visualising Results

You can visualise the actual vs. predicted Ethereum prices using the following code:

![Ethereum Price Prediction](https://github.com/user-attachments/assets/f8e69e0a-38be-49ec-b875-20329eba58a5)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![Ethereum Price Prediction After Hyperparameter Tuning](https://github.com/user-attachments/assets/5327f554-6fd6-4240-b232-a4bc176c2304)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction After Hyperparameter Tuning')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

```
![Model Loss](https://github.com/user-attachments/assets/257e2563-c503-4345-807e-9333471c1d9a)
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Ethereum Prices')
plt.plot(predicted_prices, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![model_architecture](https://github.com/user-attachments/assets/ed4d9ee9-c31e-4a1b-9c7d-6799d6363785)
```python
import matplotlib.pyplot as plt
# Plot training & validation loss values
plt.figure(figsize=(14, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

```

## Model Performance
The model's performance is evaluated using metrics such as:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted prices.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted prices.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared difference between actual and predicted prices.
- **Accuracy**: For reference, the percentage of predictions that fall within a certain error range from the actual price.

### Example Output:

```
Mean Squared Error Percentage: 0.05%
Root Mean Squared Error (RMSE): 2.54%
Mean Absolute Error (MAE): 1.80%
R-squared Percentage: 98.09%
Accuracy: 98.31288343558282%

```

## Conclusion
A Long Short-Term Memory (LSTM) model with Keras and TensorFlow is an appropriate choice for predicting Ethereum prices due to its ability to effectively model long-term dependencies and subtle patterns present in cryptocurrency data. The LSTM architecture, with its recurrent connections and gating mechanisms, can capture the intricate nuances and temporal relationships inherent in financial time series data.

This project demonstrates the use of LSTM neural networks for predicting Ethereum prices based on historical data. It showcases the complete process of loading and preprocessing data, building and training an LSTM model and evaluating its performance using various metrics.

LSTMs are particularly advantageous when dealing with long sequences of data, as they can selectively remember and forget information over extended periods, making them well-suited for tasks like cryptocurrency price prediction, where past price movements and market dynamics can influence future prices.

By leveraging the power of deep learning frameworks like Keras and TensorFlow, the LSTM model can be efficiently trained on large datasets, allowing it to learn complex patterns and make accurate predictions on unseen data.

While the performance of other models like SVR should also be evaluated, the LSTM approach with Keras and TensorFlow demonstrates promising results and aligns well with the characteristics of cryptocurrency price data, making it a strong candidate for this prediction task.

## References

- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Keras documentation](https://keras.io/)
- [Graphviz download](https://graphviz.gitlab.io/download/)
- [PyDot download](https://pypi.org/project/pydot/)
- [Pydotplus download](https://pypi.org/project/pydotplus/)
- [Matplotlib documentation](https://matplotlib.org/)
- [Pandas documentation](https://pandas.pydata.org/)
- [Numpy documentation](https://numpy.org/doc/stable/)
