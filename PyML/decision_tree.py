import numpy as np
import pandas as pd
from PyML.util_preprocess import get_spam_data, k_fold_split,print_info_table, normalize_data, plot_roc_curve_d
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

class Node(object):
    """
    This class represents a node in a decision tree.

    Attributes:
        left (Node): The left child of the node.
        right (Node): The right child of the node.
        splitting_info (dict): A dictionary that contains information about the feature and threshold that was used to split the data at this node.
        label_info (dict): A dictionary that contains information about the class labels and their counts in the data that reached this node.
        label (int): The class label that is assigned to this node if it is a leaf node.
        entropy (float): The entropy of the data that reached this node.
    """

    def __init__(self):
        """
        Initializes a new instance of the Node class.
        """
        self.left = None
        self.right = None
        self.splitting_info = None
        self.label_info = None
        self.label = None
        self.entropy = None

    def __str__(self, level=0):
        """
        Returns a string representation of the node and its subtree.

        Args:
            level (int): The level of the node in the tree.

        Returns:
            str: A string representation of the node and its subtree.
        """
        ret = "{}Label={}, LabelInfo={}, Entropy={}".format("\t" * level, self.label, self.label_info, self.entropy)

        if self.splitting_info is not None:
            ret += " Feature={}, Threshold={}".format(self.splitting_info['splitting_feature_index'],
                                                      self.splitting_info['splitting_threshold'])

        ret += "\n\n"

        if self.left is not None:
            ret += self.left.__str__(level + 1)
        if self.right is not None:
            ret += self.right.__str__(level + 1)

        return ret

    def predict_label(self, feature_vector):
        """
        Predicts the class label for the given feature vector by recursively traversing the decision tree.

        Args:
            feature_vector (array-like): A 1-D array containing the feature values for a single instance.

        Returns:
            int: The predicted class label for the given feature vector.
        """
        if self.left is None or self.right is None:
            return self.label

        splitting_threshold = self.splitting_info['splitting_threshold']
        splitting_feature_index = self.splitting_info['splitting_feature_index']
        if feature_vector[splitting_feature_index] <= splitting_threshold:
            return self.left.predict_label(feature_vector)
        else:
            return self.right.predict_label(feature_vector)



def create_tree_util(current_level, feature_vectors, labels, min_sample_size, max_depth):
    """
    Recursive helper function to create a decision tree by splitting the dataset based on the given feature vectors and labels.

    Args:
        current_level (int): The current level of the tree.
        feature_vectors (array-like): A 2-D array containing the feature vectors for each instance in the dataset.
        labels (array-like): A 1-D array containing the class labels for each instance in the dataset.
        min_sample_size (int): The minimum number of samples required to split a node.
        max_depth (int): The maximum depth of the decision tree.

    Returns:
        Node: The root node of the decision tree.
    """
    if labels.size < min_sample_size or current_level > max_depth:
        return None

    freq = Counter(labels)
    max_freq_label = None
    max_freq = -1
    for k, v in freq.items():
        if max_freq < v:
            max_freq_label = k
            max_freq = v

    root = Node()
    root.label = max_freq_label
    root.label_info = freq
    root.entropy = get_entropy(freq, len(labels))

    splitting_info = get_feature_to_split(labels, feature_vectors)
    if splitting_info is None:
        return root

    splitted_data = split_data(splitting_info, labels, feature_vectors)

    if splitted_data['left']['labels'].size == 0 or splitted_data['right']['labels'].size == 0:
        return root

    root.splitting_info = splitting_info
    root.left = create_tree_util(current_level + 1, splitted_data['left']['features'], splitted_data['left']['labels'],
                                 min_sample_size, max_depth)
    root.right = create_tree_util(current_level + 1, splitted_data['right']['features'],
                                  splitted_data['right']['labels'], min_sample_size, max_depth)

    return root


def create_tree(feature_vectors, labels, min_sample_size, max_depth):
    """
    Creates a decision tree by recursively splitting the dataset based on the given feature vectors and labels.

    Args:
        feature_vectors (array-like): A 2-D array containing the feature vectors for each instance in the dataset.
        labels (array-like): A 1-D array containing the class labels for each instance in the dataset.
        min_sample_size (int): The minimum number of samples required to split a node.
        max_depth (int): The maximum depth of the decision tree.

    Returns:
        Node: The root node of the decision tree.
    """
    return create_tree_util(1, feature_vectors, labels, min_sample_size, max_depth)


def split_data(splitting_info, labels, feature_vectors):
    """
    Splits the dataset into left and right parts based on the given splitting information.

    Args:
        splitting_info (dict): A dictionary containing information about the feature and threshold that was used to split the data.
        labels (array-like): A 1-D array containing the class labels for each instance in the dataset.
        feature_vectors (array-like): A 2-D array containing the feature vectors for each instance in the dataset.

    Returns:
        dict: A dictionary containing the left and right parts of the dataset, each with their own feature vectors and labels.
    """
    splitting_threshold = splitting_info['splitting_threshold']
    splitting_feature_index = splitting_info['splitting_feature_index']

    left_part_feature_vectors = []
    left_part_labels = []
    right_part_feature_vectors = []
    right_part_labels = []

    number_of_data_points = len(feature_vectors)

    i = 0
    while i < number_of_data_points:
        if feature_vectors[i][splitting_feature_index] <= splitting_threshold:
            left_part_feature_vectors.append(feature_vectors[i])
            left_part_labels.append(labels[i])
        else:
            right_part_feature_vectors.append(feature_vectors[i])
            right_part_labels.append(labels[i])

        i += 1

    return {
        'left': {
            'features': np.array(left_part_feature_vectors),
            'labels': np.array(left_part_labels)
        },
        'right': {
            'features': np.array(right_part_feature_vectors),
            'labels': np.array(right_part_labels)
        }
    }


def get_entropy(freq, number_of_data_points):
    """
    Calculates the entropy of a given set of frequencies.
    
    Parameters:
    freq (list): A list of integers representing the frequency of each class label.
    number_of_data_points (int): The total number of data points.
    
    Returns:
    float: The entropy value.
    """
    current_entropy = 0
    for label, count in freq.items():
        prob = count / number_of_data_points
        log_prob = np.log2(prob) if prob != 0 else 0
        current_entropy -= (prob * log_prob)

    return current_entropy


def get_feature_to_split(labels, feature_vectors):
    """
    Finds the feature that provides the maximum information gain for the given set of data points.
    
    Parameters:
    labels (list): A list of class labels for each data point.
    feature_vectors (list): A list of feature vectors for each data point.
    
    Returns:
    int: The index of the feature that provides the maximum information gain.
    """
    max_information_gain = 0
    splitting_feature_index = None
    splitting_threshold = None
    split_tuple_index = None

    number_of_data_points = len(labels)

    current_entropy = get_entropy(Counter(labels), number_of_data_points)

    feature_index = 0
    total_features = len(feature_vectors[0])

    while feature_index < total_features:
        temp = []
        i = 0
        
        for feature_vector in feature_vectors:
            temp.append((i, feature_vector[feature_index], labels[i]))
            i += 1

        temp = sorted(temp, key=lambda item: item[1])

        j = 1

        left_part = list(map(lambda item: labels[item[0]], temp[:j]))
        right_part = list(map(lambda item: labels[item[0]], temp[j:]))

        left_part_counter = Counter(left_part)
        right_part_counter = Counter(right_part)

        left_part_size = len(left_part)
        right_part_size = len(right_part)

        left_part_entropy = get_entropy(left_part_counter, left_part_size)
        right_part_entropy = get_entropy(right_part_counter, right_part_size)

        temp_entropy = (((left_part_size / number_of_data_points) * left_part_entropy) + (
                (right_part_size / number_of_data_points) * right_part_entropy))

        if temp_entropy < current_entropy:
            information_gain = current_entropy - temp_entropy

            if max_information_gain < information_gain:
                splitting_feature_index = feature_index
                splitting_threshold = feature_vectors[temp[j][0]][feature_index]
                max_information_gain = information_gain
                split_tuple_index = j

        while j < number_of_data_points - 1:
            last_element = right_part[0]
            del right_part[0]
            left_part.append(last_element)

            left_part_counter[last_element] += 1
            right_part_counter[last_element] -= 1

            left_part_size += 1
            right_part_size -= 1

            left_part_entropy = get_entropy(left_part_counter, left_part_size)
            right_part_entropy = get_entropy(right_part_counter, right_part_size)

            temp_entropy = (((left_part_size / number_of_data_points) * left_part_entropy) + (
                    (right_part_size / number_of_data_points) * right_part_entropy))

            information_gain = current_entropy - temp_entropy

            if max_information_gain < information_gain:
                splitting_feature_index = feature_index
                splitting_threshold = feature_vectors[temp[j][0]][feature_index]
                max_information_gain = information_gain
                split_tuple_index = j

            j += 1

        feature_index += 1

    splitting_info = None
    if splitting_feature_index is not None:
        splitting_info = {
            'splitting_feature_index': splitting_feature_index,
            'splitting_threshold': splitting_threshold,
            'max_information_gain': max_information_gain,
            'split_tuple_index': split_tuple_index
        }

    return splitting_info


def custom_decision_tree(splits):
    """
    Recursively creates a decision tree by finding the best feature to split on at each node.
    
    Parameters:
    splits (list): A list of tuples where each tuple contains the data points and their labels for a node.
    
    Returns:
    dict: The decision tree.
    """
    training_accuracy = []
    testing_accuracy = []
    info_table = []
    a = 1
    for split in splits:

        tree = create_tree(split['training']['features'], split['training']['labels'], 2, 10)

        i = 0
        number_of_training_data_points = split['training']['labels'].size
        training_predictions = []
        while i < number_of_training_data_points:
            training_predictions.append(tree.predict_label(split['training']['features'][i]))
            i += 1

        training_accuracy.append(accuracy_score(split['training']['labels'], training_predictions))

        i = 0
        number_of_testing_data_points = split['testing']['labels'].size
        testing_error = []
        testing_predictions = []
        while i < number_of_testing_data_points:
            testing_predictions.append(tree.predict_label(split['testing']['features'][i]))
            i += 1

        testing_accuracy.append(accuracy_score(split['testing']['labels'], testing_predictions))
        tn, fp, fn, tp = confusion_matrix(split['testing']['labels'], testing_predictions).ravel()
        fp_rate = fp / split['testing']['labels'].shape[0]
        fn_rate = fn / split['testing']['labels'].shape[0]
        error_rate = fp_rate + fn_rate
        info_table.append(["{}".format(i), fp_rate, fn_rate, error_rate])
    
        if a == 1:
            plot_roc_curve_d(split['testing']['labels'], testing_predictions, "Decision Tree Classifier")

        a += 1
    print_info_table(info_table)

    print('\n')
    print('Training Accuracy using Custom Decision Tree', np.mean(training_accuracy))
    print('Testing Accuracy using Custom Decision Tree', np.mean(testing_accuracy))

def decision_tree():
    """
    Reads the data from a file, creates a decision tree, and prints it.
    
    Returns:
    None
    """
    _data = get_spam_data()
    _data = normalize_data(_data)
    _k = 5
    _splits = k_fold_split(_k, _data, shuffle=True)
    custom_decision_tree(_splits)
        
