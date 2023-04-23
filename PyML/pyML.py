import PyML
from PyML.naive_bayes import demo_naive_bayes_with_bernoulli_features
from PyML.decision_tree import decision_tree
from PyML.logistic_regression import logistic_regression

def train_models():  
    """
    Function to train the models
    Returns: None
    """
    print("Training decision tree model")     
    decision_tree()
    print("Training naive bayes model")
    demo_naive_bayes_with_bernoulli_features()
    print("Training Logistic regression model")
    logistic_regression()

if __name__ == '__main__':
    train_models()


