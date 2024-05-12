## Predicting Ethereum Price with Python and Machine Learning

This README outlines the steps covered in a YouTube video (using Python and machine learning to predict the future price of Ethereum (ETH): [https://www.youtube.com/watch?v=HiDEAWdAif0](https://www.youtube.com/watch?v=HiDEAWdAif0)) on how to use Python libraries to predict the future price of Ethereum (ETH). Where instead of using CSV file containing the history data for ETH, thr Coinbase Pro API is used. **Disclaimer:** The video content is for educational purposes only and should not be considered professional investment advice.

### Prerequisites

* Python libraries: pandas, NumPy, scikit-learn (specifically SVC), matplotlib

### Steps

1. **Import Libraries:**

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.svm import SVR
   import matplotlib.pyplot as plt
   ```

2. **Data Upload:**

   * Upload the ETH price data in CSV format to Google Colab or Jupyter NoteBook.

3. **Load Data:**

   * Load the CSV data into a pandas DataFrame.

4. **Data Preprocessing:**

   * Set the date as the index of the DataFrame.
   * Create a new column named "future_5_day_price_forecast" by shifting the close price up by 5 days.

5. **Split Data:**

   * Split the data into training and testing sets.

6. **Model Training:**

   * Use Support Vector Regressor (SVR) to train the model on the training data.

7. **Model Evaluation:**

   * Evaluate the model on the testing data to determine its accuracy.

8. **Visualization:**

   * Plot the predicted values versus the actual values to visualize the results.
