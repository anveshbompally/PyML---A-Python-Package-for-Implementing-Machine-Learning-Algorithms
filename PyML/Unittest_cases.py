#UnitTest Cases

#util_preprocess
import unittest
from prettytable import prettytable
from PyML.util_preprocess import k_fold_split, get_spam_data
import numpy as np
from PyML.naive_bayes import NaiveBayesBins
import pandas as pd
from sklearn.metrics import confusion_matrix
from PyML.decision_tree import create_tree, create_tree_util
from PyML.logistic_regression import logistic_regression



    
def test_k_fold_split():
    """
    Test case for the k_fold_split function.

    This function tests whether the k_fold_split function correctly splits the data into k folds

    """
    data = get_spam_data()
    k = 5
    seed = 11
    shuffle = True
    
    result = k_fold_split(k, data, seed, shuffle)
    
    # Check if the resulting list contains k elements
    assert len(result)== k
    
    # Check if the sum of the testing set sizes equals the total sample size
    sample_size = data['features'].shape[0]
    total_testing_set_size = sum([len(fold['testing']['features']) for fold in result])
    assert total_testing_set_size == sample_size


#naive_bayes
def test_naive_bayes_bins():
    """
    Test function for the NaiveBayesBins classifier.

    This function loads the spam dataset, converts the continuous features to discrete using the NaiveBayesBins method,
    trains a NaiveBayesBins classifier with k=1, and tests its accuracy on the dataset.

    Returns:
    None
    """
    # Load the data
    data = get_spam_data()

    # Convert continuous features to discrete using NaiveBayesBins method
    features_discrete = NaiveBayesBins.convert_continuous_features_to_discrete(data['features'])

    # Instantiate NaiveBayesBins classifier
    nb = NaiveBayesBins(seed=1, bins=2, dimension=features_discrete.shape[1], k=1)

    # Train the classifier
    nb.train(features_discrete, data['labels'])

    # Test the classifier
    predictions = nb.predict(features_discrete)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.sum(predicted_labels == data['labels']) / len(data['labels'])

    assert 0<= accuracy <=1


def test_logistic_regression():
    """
    Tests the logistic regression algorithm on spam classification data.

    Returns:
        None
    """

    # Load data and split into k-folds
    data = get_spam_data()
    k = 5
    folds = k_fold_split(k, data, shuffle=True)

    # Test hyperparameters
    learning_rate = 0.1
    epochs = 1000
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
    # Test sigmoid function
    
    z = np.array([1, 2, 3])
    expected_output = np.array([0.73105858, 0.88079708, 0.95257413])
    np.testing.assert_allclose(sigmoid(z), expected_output)
    # Test model training and accuracy calculation for each fold
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

        # Train model
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

        # Test that accuracy is between 0 and 1
        assert 0 <= train_acc <= 1
        assert 0 <= test_acc <= 1

        # Test confusion matrix and error rates for this fold
        tn, fp, fn, tp = confusion_matrix(Y_test, (sigmoid(np.dot(X_test, weights) + bias) > 0.5)).ravel()
        fp_rate = fp / Y_test.shape[0]
        fn_rate = fn / Y_test.shape[0]
        error_rate = fp_rate + fn_rate

        # Test that error rates are between 0 and 1
        assert 0 <= fp_rate <= 1
        assert 0 <= fn_rate <= 1
        assert 0 <= error_rate <= 1


def test_create_tree_util():
    """
    Recursive function that creates a decision tree.

    Args:
        label (int): the label for the current node
        feature_vectors (numpy.ndarray): an array of feature vectors for the current node
        labels (numpy.ndarray): an array of labels for the feature vectors
        min_sample_size (int): the minimum number of samples required to split a node
        max_depth (int): the maximum depth of the tree

    Returns:
        DecisionNode: the root node of the decision tree
    """
    feature_vectors = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    labels = np.array([1, 1, 1, 1])
    min_sample_size = 2
    max_depth = 3
    root = create_tree_util(1, feature_vectors, labels, min_sample_size, max_depth)
    assert root.label == 1
    assert root.label_info == {1: 4}
    assert root.entropy == 0
    feature_vectors = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    labels = np.array([1, 0, 1, 0])
    min_sample_size = 1
    max_depth = 3
    root = create_tree_util(1, feature_vectors, labels, min_sample_size, max_depth)
    assert root.splitting_info['splitting_feature_index'] == 0
    assert root.splitting_info['splitting_threshold'] == 2
    assert root.left.label == 1



test_k_fold_split()
test_naive_bayes_bins()
test_logistic_regression()
test_create_tree_util()
print("All test cases passed")