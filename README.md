## Sign Language Letter Recognition Project

The aim of the project was to create a dataset of sign language letters using a special program, process the data, and train a classifier model. To improve the classifier's accuracy, a column describing which hand was used to make the letter was encoded. Data normalization was performed using the MinMaxScaler and StandardScaler functions. The dataset was divided into training and testing sets.

The test set for evaluating the model's quality was created using the `train_test_split` function in an 80/20 ratio. A classifier with multiple neural layers (MLPClassifier) was used for classification. This model optimizes the loss function using the LBFGS algorithm. The sigmoid function was used as the activation function, with a maximum of 30,000 iterations.

The model was also analyzed for hyperparameters and after removing outliers. The GridSearch and HalvingGridSearch functions did not improve classification effectiveness. Probability calibration was also used for training, using regression with the sigmoid function.

On the test set, the classifier achieved 95.5% effectiveness, measured by the f1_score metric.
