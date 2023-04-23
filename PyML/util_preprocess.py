# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Reads the spam dataset and returns features and labels as a dictionary
def get_spam_data():
    """
    Reads the spam dataset and returns features and labels as a dictionary.

    Returns:
    - A dictionary containing the 'features' and 'labels' arrays.
    """
    data = pd.read_csv('spambase.data', header=None)  # read csv file using pandas
    features = np.array(data.iloc[:, :-1])  # select all columns except the last one as features
    labels = np.array(data.iloc[:, -1])  # select the last column as labels

    # Checks if the number of feature tuples is the same as the number of label tuples
    if features.shape[0] != labels.shape[0]:
        raise Exception("Mismatch in Feature Tuples(%d) and Label Tuples(%d)" % (features.size, labels.size))

    return {
        'features': features,
        'labels': labels
    }

# Performs k-fold cross-validation by splitting the data into k-folds
def k_fold_split(k, data, seed=11, shuffle=False):
    """
    Performs k-fold cross-validation by splitting the data into k-folds.

    Args:
    - k: Integer value representing the number of folds.
    - data: A dictionary containing 'features' and 'labels' arrays.
    - seed: Integer value representing the random seed (default: 11).
    - shuffle: Boolean value representing whether to shuffle the data (default: False).

    Returns:
    - A list containing k dictionaries, each containing the training and testing data for a fold.
    """
    sample_size = data['features'].shape[0] 

    indices = np.arange(0, sample_size) 

    if shuffle:  # if shuffle is True, shuffle the indices randomly
        np.random.seed(seed)
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)  # split the indices array into k-folds
    testing_fold_index = 0  
    final_data = [] 

    # For each fold, select the corresponding testing fold and combine the remaining folds to create the training fold
    for i in range(0, k):
        training_folds = [folds[j] for j in range(0, k) if j != testing_fold_index]  # select all folds except the testing fold
        training_data_indices = np.concatenate(training_folds) 
        training_data_features = data['features'][training_data_indices]  
        training_data_labels = data['labels'][training_data_indices]  

        testing_data_indices = folds[testing_fold_index]  # select the indices of the testing fold
        testing_data_features = data['features'][testing_data_indices]  
        testing_data_labels = data['labels'][testing_data_indices] 

        # Store the training and testing data for this fold as a dictionary
        temp = {
            'training': {
                'features': training_data_features,
                'labels': training_data_labels
            },
            'testing': {
                'features': testing_data_features,
                'labels': testing_data_labels
            }
        }


        final_data.append(temp)
        testing_fold_index += 1

    return final_data

# Define a function that prints a table summarizing the performance of a model on different folds
def print_info_table(info_table):
    """
    Prints a pretty table summarizing the performance of a model on different folds.

    Args:
    - info_table: A list of lists containing fold-wise performance metrics.

    Returns:
    - None
    """
    
    # Define the column headings of the table
    table = PrettyTable(["Fold", "FP Rate", "FN Rate", "Error Rate"])
    
    # Calculate the average values of the metrics across all the folds
    avg_fp_rate = np.mean([info[1] for info in info_table])
    avg_fn_rate = np.mean([info[2] for info in info_table])
    avg_error_rate = np.mean([info[3] for info in info_table])

    # Append the average metric values as a new row to the info_table
    info_table.append(["Avg", avg_fp_rate, avg_fn_rate, avg_error_rate])

    # Populate the table with the metric values for each fold
    for info in info_table:
        table.add_row(info)

    # Print the table
    print(table)

# Define a function that plots the ROC curve for the given testing data
def plot_roc_curve(testing_true_labels, testing_predictions, title):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for the given testing data.

    Parameters:
    - testing_true_labels (array-like): A 1D array-like object of shape (n_samples,) that contains the true labels for testing data.
    - testing_predictions (array-like): A 2D array-like object of shape (n_samples, 2) that contains the predicted probabilities for testing data.
    - title (str): The title of the ROC curve plot.

    Returns:
    - None
    """

    # Create a copy of the testing_predictions array
    testing_predictions = testing_predictions.copy()

    # Calculate the log-odds for the predicted probabilities
    log_odds = np.log(testing_predictions[:, 1]) - np.log(testing_predictions[:, 0])

    # Sort the indices of the testing_true_labels array based on the log-odds values
    sorted_indices = np.argsort(log_odds)

    # Compute the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(testing_true_labels[sorted_indices], np.sort(log_odds))
    area_under_curve = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % area_under_curve)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
# define a function to normalize the feature vectors of the input data using Z-score normalization
def normalize_data(data):
    # extract the feature vectors from the data dictionary
    feature_vectors = data['features']

    # get the number of features in the data
    total_features = len(feature_vectors[0])
    
    # loop through each feature
    i = 0
    while i < total_features:
        # extract the values for the current feature
        feature_values = feature_vectors[:, i]
        # calculate the mean and standard deviation of the feature values
        mean = feature_values.mean()
        std = feature_values.std()
        # normalize the feature values using Z-score normalization
        normalized_values = (feature_values - mean) / std
        # replace the original feature values with the normalized values
        feature_vectors[:, i] = normalized_values
        # move to the next feature
        i += 1

    # update the 'features' key in the data dictionary with the normalized feature vectors
    data['features'] = feature_vectors
    # return the updated data dictionary
    return data

# define a function to plot the ROC curve for binary classification
def plot_roc_curve_d(testing_true_labels, testing_predictions, title):
    # make a copy of the predicted probabilities
    testing_predictions = testing_predictions.copy()

    # calculate the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(testing_true_labels, testing_predictions)
    # calculate the area under the ROC curve
    area_under_curve = auc(fpr, tpr)

    # plot the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % area_under_curve)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
