# Building a Recurring Neural Network(RNN) Long Short Term Memory(LSTM) for predicting gold closing prices using time-series data.
# Instructions


Tools: Python, Pandas, Numpy, matplotlib, tensorflow, sklearn, Nasdaq Account, Quandl API



## Initial Setup: 
To ensure models' reproducibility, set the random seed for numpy and tensorflow libraries.

## Data Preparation
In this section, you will retrieve the Gold historical prices from the London Bullion Market Association using the Quandl API. Be sure to have your Quandl API key at hand.

## Data Retrieval
1) Import your Quandl API key from an environment variable named quandl_key.
2) Open the "Gold Price: London Fixing" at the Quandl website, and set the URL to retrieve the historical prices of gold in json format.
3) Use the requests library to retrieve the historical prices of gold in json format.
4) Explore the json response data and create a Pandas DataFrame containing the historical prices of gold in all the different currencies provided by the API, and set the data as the index of the DataFrame.

Note: Remember that while working with time-series data, it's important to transform dates data to datetime data type explicitly. Review the Quandl API documentation for time-series data if needed.

## Data Cleaning
Before you continue, corroborate if there are any null or missing values in the DataFrame, if so, fill the missing values with the previous price in the series.
Note: You may want to review the Working with missing data guide from the Pandas documentation.

## Create the Features X and Target y Data
Use the window_data() function below to create the features set X and the target vector y. Define a window size of 30 days and use the column of the closing gold price (USD (PM)) for the feature and target column; this will allow your model to predict gold prices in USD.

def window_data(df, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i : (i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

## Split Data Between Training and Testing Sets
To avoid the dataset being randomized, manually create the training and testing sets using array slicing. Use 70% of the data for training and the remainder for testing.

## Scale Data with MinMaxScaler
Before training the RNN LSTM model, use the MinMaxScaler from sklearn to scale the training and testing data between 0 and 1.
Note: You need to scale both features and target sets.

## Reshape Features Data for the LSTM Model
The LSTM API from Keras needs to receive the features data as a vertical vector, so you must reshape the X data in the form reshape((X_train.shape[0], X_train.shape[1], 1)). Both sets, training, and testing should be reshaped.

## Build and Train the LSTM RNN
In this section, you will design a custom LSTM RNN in Keras and fit (train) it using the training data we defined.
You will need to:
1) Define the model architecture in Keras.
2) Compile the model.
3) Fit the model with the training data.

## Create the LSTM RNN Model Structure
Design the structure of your RNN LSTM as follows:
- Number of units per layer: 30 (same as the window size)
- Dropout fraction: 0.2 (20% of neurons will be randomly dropped on each epoch)
- Add three LSTM layers to your model, remember to add a Dropout layer after each LSTM layer, and to set return_sequences=True in the first two layers only.
- Add a Dense output layer with one unit.

## Compile the LSTM RNN Model
Compile the model using the adam optimizer, and mean_square_error as loss function since the value you want to predict is continuous.

## Train the Model
Train (fit) the model with the training data using 10 epochs and a batch_size=90. Since you are working with time-series data, remember to set shuffle=False since it's necessary to keep the sequential order of the data.

## Model Performance
In this section, you will evaluate the model using the test data.
You will need to:
1) Evaluate the model using the X_test and y_test data.
2) Use the X_test data to make predictions.
3) Create a DataFrame of real (y_test) vs. predicted values.
4) Plot the real vs. predicted values as a line chart.

## Evaluate the Model
Use the evaluate() method of the model using the testing data.

## Make Predictions
- Use the predict() method of the model to make some closing gold price predictions using your brand new LSTM RNN model and your testing data. Save the predictions in a variable called predicted.

- Since you scaled the original values using the MinMaxScaler, you need to recover the original gold prices to better understand of the predictions. Use the inverse_transform() method of the scaler to decode the scaled testing and predicted values to their original scale.

## Plotting Predicted Vs. Real Prices
Create a Pandas DataFrame with an index and two columns as follows to plot the predicted vs. the actual gold prices.
- Index: The datetime index from the DataFrame created previously.
- Column 1: Actual prices (testing data)
- Column 2: Predicted prices
- Use the plot() method from the DataFrame to create a line chart to contrast the actual vs. the predicted gold prices.

<img width="572" alt="Screen Shot 2022-07-05 at 12 41 15 PM" src="https://user-images.githubusercontent.com/83780964/177376128-f935e691-a1ad-4c04-9953-cc5d6e1188eb.png">

# Neural Networks
Neural networks are a set of algorithms that are modeled after the human brain. They’re designed to recognize patterns and interpret sensory data through a kind of machine perception, labeling, or clustering raw input.

This power of our brain to process information and make predictions and interpretations inspired neurophysiologists and mathematicians to start the development of artificial neural networks (ANN).

<img width="1053" alt="Screen Shot 2022-07-04 at 3 03 13 PM" src="https://user-images.githubusercontent.com/83780964/177206852-387be43a-10e1-47a0-b7af-9c9ef77261f9.png">

## Deep Learning 
Deep learning models are neural networks with more than one hidden layer.

<img width="747" alt="Screen Shot 2022-07-05 at 11 48 31 AM" src="https://user-images.githubusercontent.com/83780964/177367303-9c3b9a79-cf38-4f9e-af27-e384a9a21989.png">

Neural networks work by calculating the weights of various input data and passing them on to the next layer of neurons. The number of layers that are included in a neural network model determines whether it is a deep learning model or not. In general, networks with more than one "hidden" layer can be classified as "deep."

## Introducing the ROC Curve and AUC
We use the confusion matrix to assess the performance of a binary classification model.

<img width="993" alt="Screen Shot 2022-07-05 at 11 51 04 AM" src="https://user-images.githubusercontent.com/83780964/177367751-2d15b1b7-d041-4c09-b5da-e108cee28a3b.png">

<img width="1156" alt="Screen Shot 2022-07-05 at 11 52 05 AM" src="https://user-images.githubusercontent.com/83780964/177367897-e83c4133-a3a8-4df5-8c05-453ecbeb28d3.png">

The ROC curve and AUC are techniques that allow us to check and visualize the performance of a classification model.

<img width="545" alt="Screen Shot 2022-07-05 at 11 52 45 AM" src="https://user-images.githubusercontent.com/83780964/177368016-598a0637-697d-46c3-b710-021a85e15fe9.png">

## Understanding ROC

ROC stands for "Receiver Operating Characteristic". The ROC curve shows the performance of a classification model as its discrimination threshold is varied.

<img width="519" alt="Screen Shot 2022-07-05 at 11 53 52 AM" src="https://user-images.githubusercontent.com/83780964/177368183-26260304-c5c2-49ca-89c4-7e69826c5c9c.png">

To plot a ROC curve, we use two parameters:

<img width="922" alt="Screen Shot 2022-07-05 at 11 54 26 AM" src="https://user-images.githubusercontent.com/83780964/177368306-76dfbd50-aff9-40f0-b4ed-55dbbeac8dcf.png">

Every point in the ROC curve represents the TPR vs. FPR at different thresholds. 
This image shows a typical ROC curve.

<img width="544" alt="Screen Shot 2022-07-05 at 11 55 26 AM" src="https://user-images.githubusercontent.com/83780964/177368463-ce6cf441-5829-412b-9241-f591468e99a3.png">

## Understanding AUC

AUC stands for Area Under the ROC Curve.

Interpreting the ROC curve can be challenging. Fortunately, we have the AUC that measures the area below the entire ROC curve from (0,0) to (1,1).

<img width="520" alt="Screen Shot 2022-07-05 at 11 57 48 AM" src="https://user-images.githubusercontent.com/83780964/177368945-21401c5f-7208-4f55-a260-fa6351fe3cf8.png">

The value of AUC ranges from 0 to 1.

 AUC = 1  is a paradox: it indicates that the model is perfect, but you may not trust your model because it might be overfitted.
 
 <img width="572" alt="Screen Shot 2022-07-05 at 11 59 08 AM" src="https://user-images.githubusercontent.com/83780964/177369219-e703e842-e029-4d41-864d-bbbefedc4b6f.png">

 AUC = 0.50  means that the model is unable to distinguish between positive and negative classes.
 
 <img width="577" alt="Screen Shot 2022-07-05 at 11 59 45 AM" src="https://user-images.githubusercontent.com/83780964/177369366-ae467f5e-9d4e-406e-bb8c-aca428ab071b.png">

Ideally, we want AUC values ranging between 0 and 1. The higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s.

<img width="1043" alt="Screen Shot 2022-07-05 at 12 00 23 PM" src="https://user-images.githubusercontent.com/83780964/177369489-148aad4b-ff86-4f33-97b4-19e75df23423.png">

So, a model with an AUC=0.90 may be better than a model with an AUC=0.65.

## Recurrent Neural Networks (RNNs) 
RNNs are able to remember the past, and their decisions are influenced by what they have learned in the past.

## ANNs vs. RNNs
We can use ANNs to identify the type of car from a still image. But can we predict the direction of a car in movement?

<img width="1126" alt="Screen Shot 2022-07-05 at 12 02 38 PM" src="https://user-images.githubusercontent.com/83780964/177369866-1e63dd81-1d4e-4020-93db-8f58788d0118.png">

RNNs are good at modeling sequence data thanks to their sequential memory. Using RNNs, we can predict that the car is moving to the right.

<img width="1076" alt="Screen Shot 2022-07-05 at 12 03 25 PM" src="https://user-images.githubusercontent.com/83780964/177369970-865e7542-1bbf-44ce-ba79-b53e62ce17f1.png">

RNNs are good at modeling sequence data thanks to their sequential memory. Using RNNs, we can predict that the car is moving to the right.

<img width="976" alt="Screen Shot 2022-07-05 at 12 04 00 PM" src="https://user-images.githubusercontent.com/83780964/177370079-14e8f0e3-194f-4166-a491-8921e150b740.png">

## How Do RNNs Work?
<img width="1074" alt="Screen Shot 2022-07-05 at 12 05 43 PM" src="https://user-images.githubusercontent.com/83780964/177370437-b63c22c6-8c0e-4e14-8b91-569886896f93.png">

The sentence is split into individual words. RNNs work sequentially, so we feed it one word at a time. By the final step, the RNN has encoded information from all the words in previous steps. 
<img width="1067" alt="Screen Shot 2022-07-05 at 12 08 25 PM" src="https://user-images.githubusercontent.com/83780964/177370847-7b4476a7-3739-4e00-9f4a-78a1d0434d3b.png">

<img width="1098" alt="Screen Shot 2022-07-05 at 12 09 26 PM" src="https://user-images.githubusercontent.com/83780964/177371017-b778bcbc-0092-4638-a9f7-6fbce2be763d.png">

## RNNs Are Forgetful
RNNs only “remember” the most recent few steps. 

## Long Short-Term Memory (LSTM)
LSTMs to the Rescue. LSTM (Long Short-Term Memory) RNNs are one solution for longer time windows. An LSTM RNN works like an original RNN, but it selects which types of longer-term events are worth remembering and which are okay to forget.

<img width="779" alt="Screen Shot 2022-07-05 at 12 11 57 PM" src="https://user-images.githubusercontent.com/83780964/177371425-5c594d4c-4f74-42b9-9f56-a0953fe09699.png">

## Dropout
Dropout consists of removing units from the hidden layers by randomly selecting a fraction of the hidden nodes and setting their output to zero, regardless of the input. A different subset of units is randomly selected every time we feed a training example.

<img width="1102" alt="Screen Shot 2022-07-05 at 12 13 33 PM" src="https://user-images.githubusercontent.com/83780964/177371696-e5892947-e998-4602-b9d7-87f1b43008e4.png">
