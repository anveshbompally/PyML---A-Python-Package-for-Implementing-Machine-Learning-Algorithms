import numpy as np
import pandas as pd
import math
from PyML.util_preprocess import get_spam_data, k_fold_split, plot_roc_curve_d, print_info_table
from sklearn.metrics import confusion_matrix

def logistic_regression():
    """
    Trains a logistic regression model using k-fold cross-validation on the spam dataset.

    Returns:
    None
    """

    # Load data and split into k-folds
    data = get_spam_data()
    k = 5
    folds = k_fold_split(k, data, shuffle=True)

    # Set hyperparameters
    learning_rate = 0.1
    epochs = 1000
    calc_table = []
    info_table = []

    # Sigmoid function
    def sigmoid(z):
        """
        Computes the sigmoid function on input z.

        Args:
        z (numpy.ndarray): Input to the sigmoid function.

        Returns:
        numpy.ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    a = 1

    # Train model
    for fold in range(k):
        # Split data into training and testing sets for this fold
        X_train = folds[fold]['training']['features']
        Y_train = folds[fold]['training']['labels']
        X_test = folds[fold]['testing']['features']
        Y_test = folds[fold]['testing']['labels']

        # Normalize data
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Initialize weights and bias
        weights = np.zeros(X_train.shape[1])
        bias = 0

        for epoch in range(epochs):
            for i in range(len(X_train)):
                # Calculate prediction and error
                z = np.dot(weights, X_train[i]) + bias
                prediction = sigmoid(z)
                error = prediction - Y_train[i]

                # Update weights and bias
                weights -= learning_rate * error * X_train[i]
                bias -= learning_rate * error

            # Calculate accuracy on training set
            train_acc = np.mean((sigmoid(np.dot(X_train, weights) + bias) > 0.5) == Y_train)

            # Calculate accuracy on testing set
            test_acc = np.mean((sigmoid(np.dot(X_test, weights) + bias) > 0.5) == Y_test)

            # Store info for this epoch
            calc_table.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'test_acc': test_acc
            })

        # Calculate confusion matrix and error rates for this fold
        tn, fp, fn, tp = confusion_matrix(Y_test, (sigmoid(np.dot(X_test, weights) + bias) > 0.5)).ravel()
        fp_rate = fp / Y_test.shape[0]
        fn_rate = fn / Y_test.shape[0]
        error_rate = fp_rate + fn_rate

        # Store info for this fold
        info_table.append(["{}".format(i), fp_rate, fn_rate, error_rate])

        # Plot ROC curve for the first fold
        if a == 1:
            plot_roc_curve_d(Y_test, (sigmoid(np.dot(X_test, weights) + bias) > 0.5), "Logistic Regression")
        
        a += 1

    # Print table of error rates for each fold
    print_info_table(info_table)
    train_acc_list = [info['train_acc'] for info in calc_table]
    test_acc_list = [info['test_acc'] for info in calc_table]
    avg_train_acc = sum(train_acc_list) / len(train_acc_list)
    avg_test_acc = sum(test_acc_list) / len(test_acc_list)
    print("Average Training Accuracy:", avg_train_acc)
    print("Average Testing Accuracy:", avg_test_acc)