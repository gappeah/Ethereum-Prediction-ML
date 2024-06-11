# Predicting Ethereum Price with Python and Machine Learning

This project aims to analyze and predict Ethereum prices using two different machine learning approaches: Support Vector Machine (SVM) with Scikit-learn and Long Short-Term Memory (LSTM) networks with TensorFlow and Keras.
By comparing these approaches, we can understand their effectiveness in handling time series data and their predictive capabilities for cryptocurrency prices.
This repository contains two ipynb files demonstrating how to analyze and predict the price of Ethereum using Python and various machine-learning techniques. The dataset used for this analysis spans from August 7, 2015, to June 2, 2024.

### Machine Learning Models

#### 1. Support Vector Machine (SVM) with Scikit-learn
- **Preprocessing**:
  - Data is normalized using `MinMaxScaler`.
  - Missing values are handled.
- **Hyperparameter Tuning**:
  - Hyperparameters such as `C`, `kernel`, and `gamma` are tuned using `GridSearchCV`.
- **Model Evaluation**:
  - The model is evaluated using MSE and R² metrics.
- **Advantages**:
  - SVM is effective in high-dimensional spaces.
  - SVM is memory efficient.
- **Disadvantages**:
  - SVMs are less effective when the number of features exceeds the number of samples.
  - SVMs do not perform well with large datasets.

#### 2. Long Short-Term Memory (LSTM) with TensorFlow and Keras
- **Preprocessing**:
  - Data is normalized using `MinMaxScaler`.
  - Missing values are handled.
- **Model Architecture**:
  - An LSTM network is constructed with multiple layers, including dropout for regularization.
- **Hyperparameter Tuning**:
  - Hyperparameters such as the number of LSTM units, dropout rates, and batch size are manually adjusted.
- **Model Evaluation**:
  - The model is evaluated using MSE and R² metrics.
- **Advantages**:
  - LSTMs are effective for time series prediction due to their ability to capture temporal dependencies.
  - LSTMs can handle long-term dependencies.
- **Disadvantages**:
  - LSTMs require more computational resources.
  - LSTMs can be complex to tune.

### Results and Comparison
- **Performance Metrics**:
  - The performance of each model is compared using MSE and R² metrics.
  - The results indicate that LSTM outperforms SVM in capturing the temporal dependencies of the Ethereum price data.
- **Model Visualization**:
  - The architecture of the LSTM model is visualized to understand the structure and complexity.
  - Training and validation loss plots are used to analyze the model's learning process.

### Work in Progress
- Further tuning and optimization of hyperparameters for both models.
- Exploration of additional preprocessing techniques.
- Incorporation of other evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Implementation of additional deep learning models for comparison.

## Getting Started

### Prerequisites
- Python 3.x
- Scikit-learn
- TensorFlow
- Keras
- Pandas
- Numpy
- Matplotlib
- Graphviz (for model visualization)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/gappeah/Ethereum-Prediction-ML.git
    cd ethereum-price-prediction
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. Preprocess the data and handle missing values.
2. Normalize the features using `MinMaxScaler`.
3. Train and evaluate the SVM model using Scikit-learn:
    ```python
    python main_svm.ipynb
    ```
4. Train and evaluate the LSTM model using TensorFlow and Keras:
    ```python
    python main_lstm_copy.ipynb
    ```
5. Visualize the results and compare the performance metrics.

### Results
- Visualizations of the model architectures.
- Plots comparing the actual and predicted Ethereum prices.
- Performance metrics (MSE and R²) for both models.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or additions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## References

- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [Keras documentation](https://keras.io/)
- [Graphviz download](https://graphviz.gitlab.io/download/)
