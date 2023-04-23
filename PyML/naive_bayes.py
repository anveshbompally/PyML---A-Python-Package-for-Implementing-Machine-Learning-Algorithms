import PyML
from PyML.util_preprocess import get_spam_data, k_fold_split, plot_roc_curve,print_info_table
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class NaiveBayesBins:
    """
    A Naive Bayes classifier using bins to discretize continuous features.
    """

    def __init__(self, seed, bins, dimension, k) -> None:
        """
        Constructor method to initialize the NaiveBayesBins object.

        :param seed: The seed for the random number generator.
        :type seed: int
        :param bins: The number of bins to use for discretizing continuous features.
        :type bins: int
        :param dimension: The number of features in the dataset.
        :type dimension: int
        :param k: The smoothing parameter for the Naive Bayes classifier.
        :type k: int
        """
        super().__init__()
        self.dimension = dimension
        self.bins = bins
        self.k = k
        self.seed = seed
        self.prior_p_non_spam = None
        self.prior_p_spam = None
        self.non_spam_bin_prob = None
        self.spam_bin_prob = None

    def train(self, features, labels):
        """
        Trains the Naive Bayes classifier using the given features and labels.

        :param features: The features of the dataset.
        :type features: numpy array
        :param labels: The labels of the dataset.
        :type labels: numpy array
        """
        non_spam_count = labels[labels == 0].shape[0]
        spam_count = labels[labels == 1].shape[0]

        self.prior_p_non_spam = non_spam_count / features.shape[0]
        self.prior_p_spam = spam_count / features.shape[0]

        non_spam_bin_prob = []
        spam_bin_prob = []

        for i in range(self.dimension):
            non_spam_prob = []
            spam_prob = []
            for j in range(self.bins):
                f_i = features[:, i][labels == 0]
                i_bin_j_non_spam_count = f_i[f_i == j].shape[0]
                p_i_bin_j_non_spam = (i_bin_j_non_spam_count + self.k) / (non_spam_count + (2 * self.k))
                non_spam_prob.append(p_i_bin_j_non_spam)
                f_i = features[:, i][labels == 1]
                i_bin_j_spam_count = f_i[f_i == j].shape[0]
                p_i_bin_j_spam = (i_bin_j_spam_count + self.k) / (spam_count + (2 * self.k))
                spam_prob.append(p_i_bin_j_spam)

            non_spam_bin_prob.append(non_spam_prob)
            spam_bin_prob.append(spam_prob)

        self.non_spam_bin_prob = np.array(non_spam_bin_prob)
        self.spam_bin_prob = np.array(spam_bin_prob)


    def predict(self, features):

        predicted_non_spam_probs = []
        predicted_spam_probs = []

        for f in features:

            p_non_spam = 1
            p_spam = 1
            for j in range(self.bins):
                p_non_spam *= np.product(self.non_spam_bin_prob[f == j, j])
                p_spam *= np.product(self.spam_bin_prob[f == j, j])

            p_non_spam *= self.prior_p_non_spam
            p_spam *= self.prior_p_spam

            predicted_non_spam_probs.append(p_non_spam)
            predicted_spam_probs.append(p_spam)

        return np.column_stack((predicted_non_spam_probs, predicted_spam_probs))

    @staticmethod
    def convert_continuous_features_to_discrete(features):
        """
        Converts continuous features to discrete by binning them based on their mean value.

        :param features: The continuous features to be converted to discrete.
        :type features: numpy array
        :return: The discretized features.
        :rtype: numpy array
        """
        features = features.copy()
        mean = features.mean(axis=0)

        dimension = features.shape[1]
        for i in range(dimension):
            f_i = features[:, i]
            f_i[f_i <= mean[i]] = 0
            f_i[f_i > mean[i]] = 1

        return features

def demo_classifier(data, classifier, classifier_name):
    """
    Trains and evaluates a classifier on a given dataset using k-fold cross-validation.
    Prints the error rate, false positive rate, and false negative rate for each fold and overall accuracy.

    Args:
    data: A dictionary with 'features' and 'labels' keys, representing the dataset.
    classifier: An object that has 'train(features, labels)' and 'predict(features)' methods.
    classifier_name: A string representing the name of the classifier.

    Returns:
    None
    """
    k_folds = k_fold_split(10, data, seed=34928731, shuffle=True)

    training_accuracy = []
    testing_accuracy = []

    info_table = []
    i = 1
    for k_fold_data in k_folds:
        classifier.train(k_fold_data['training']['features'], k_fold_data['training']['labels'])

        training_predictions = classifier.predict(k_fold_data['training']['features'])
        training_prediction_labels = np.argmax(training_predictions, axis=1)

        testing_predictions = classifier.predict(k_fold_data['testing']['features'])
        testing_prediction_labels = np.argmax(testing_predictions, axis=1)

        training_accuracy.append(
            accuracy_score(k_fold_data['training']['labels'], training_prediction_labels))

        testing_true_labels = k_fold_data['testing']['labels']
        testing_accuracy.append(accuracy_score(testing_true_labels, testing_prediction_labels))

        tn, fp, fn, tp = confusion_matrix(testing_true_labels, testing_prediction_labels).ravel()
        fp_rate = fp / testing_true_labels.shape[0]
        fn_rate = fn / testing_true_labels.shape[0]
        error_rate = fp_rate + fn_rate
        info_table.append(["{}".format(i), fp_rate, fn_rate, error_rate])

        if i == 1:
            plot_roc_curve(testing_true_labels, testing_predictions, classifier_name)

        i += 1

    print_info_table(info_table)

    print("Training accuracy: ", np.mean(training_accuracy))
    print("Testing accuracy: ", np.mean(testing_accuracy))
    print()


def demo_naive_bayes_with_bernoulli_features():
    """
    Trains and evaluates a Naive Bayes classifier on spam dataset with binarized features using k-fold cross-validation.
    Prints the false positive rate, false negative rate, and error rate for each fold and overall accuracy.

    Args:
    None

    Returns:
    None
    """
    data = get_spam_data()
    data['features'] = NaiveBayesBins.convert_continuous_features_to_discrete(data['features'])
    demo_classifier(data, NaiveBayesBins(1, 2, data['features'].shape[1], 1), "Naive Bayes Bernoulli")
