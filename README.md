# Heart Disease Artificial Neural Network (ANN) Using PyTorch
This project implements an Artificial Neural Network (ANN) using PyTorch for predicting heart disease. The ANN is trained on a dataset with roughly 1000 data points and has been evaluated to achieve an accuracy of 87% on unseen data using input features such as age, sex, 	chest pain type, etc.

## Preprocessing 

The heart disease dataset is preprocessed using the preprocessData() function. The function performs the following steps:

- Reads the heart disease dataset from the CSV file "heart.csv".
- Maps categorical variables to numerical values using the map() function.
- Splits the dataset into train and test data using 80% of the data for training and 20% for testing.
- Converts the train and test data to torch tensors using torch.tensor().
- Creates DataLoader objects for loading train and test data in batches using torch.utils.data.DataLoader().

Dataset obtained from: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Neural Network Architecture 

The neural network architecture is defined in the HeartDiseaseANN class. The class takes the following arguments during initialisation:

input_size: Size of the input features.
output_size: Size of the output predictions.
hidden_layers: List of integers specifying the sizes of hidden layers.
learning_rate: Learning rate for optimisation.
The neural network has an input layer, hidden layers, and an output layer. ReLU activation function is applied to the input and hidden layers, and sigmoid activation function is applied to the output layer.

The neural network model is trained using the trainModel() method, which takes the following arguments:

- dataloader: DataLoader for loading training data.
- epochs: Number of epochs for training.
- criterion: Loss function for optimisation.
- optimiser: Optimisation algorithm for updating model parameters.

## Packages Used

Pandas, PyTorch, MatPlotLib, NumPy
