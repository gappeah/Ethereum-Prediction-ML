# Predicting Ethereum Price with Python and Machine Learning

This repository contains a Jupyter notebook that demonstrates how to analyze and predict the price of Ethereum using Python and various machine learning techniques. The dataset used for this analysis spans from August 7, 2015, to June 2, 2024.

## Files in the Repository

- `main.ipynb`: Jupyter notebook containing the data analysis, visualization, and machine learning steps for predicting Ethereum prices.
- `ethereum_2015-08-07_2024-06-02.csv`: CSV file containing Ethereum price data, including columns for `Start`, `End`, `Open`, `High`, `Low`, `Close`, `Volume`, and `Market Cap`.

## Overview

### Dataset

The dataset includes the following columns:
- `Start`: The start date of the data entry.
- `End`: The end date of the data entry.
- `Open`: The opening price of Ethereum on the start date.
- `High`: The highest price of Ethereum on the start date.
- `Low`: The lowest price of Ethereum on the start date.
- `Close`: The closing price of Ethereum on the start date.
- `Volume`: The trading volume of Ethereum on the start date.
- `Market Cap`: The market capitalization of Ethereum on the start date.

### Jupyter Notebook (`main.ipynb`)

The notebook is structured as follows:
1. **Introduction**: Description of the project and its objectives.
2. **Data Loading and Preprocessing**: 
   - Importing necessary libraries.
   - Loading the Ethereum price data from the CSV file.
   - Displaying the first few rows of the dataset.
3. **Data Analysis and Visualization**: 
   - Exploratory data analysis to understand the trends and patterns in the data.
   - Visualizing the historical prices and other relevant metrics.
4. **Feature Engineering**: Creating additional features from the raw data to improve the performance of machine learning models.
5. **Model Building and Evaluation**: 
   - Building machine learning models to predict Ethereum prices.
   - Evaluating the performance of the models using appropriate metrics.
6. **Conclusion**: Summarizing the findings and potential future work.

## Getting Started

### Prerequisites

To run the notebook, you need to have the following libraries installed:
- `matplotlib`
- `datetime`
- `seaborn`
- `numpy`
- `re`
- `mplfinance`
- `pandas_datareader`
- `pandas`

You can install these libraries using `pip`:
```sh
pip install matplotlib seaborn numpy mplfinance pandas_datareader pandas
