# PyML---A-Python-Package-for-Implementing-Machine-Learning-Algorithms

The PyML package is aimed at implementing machine learning algorithms for a classification problem into a single function so that three of these models can be compared at a time and choose the best one suited based on the scores. Specifically, we will be implementing logistic regression, Naive Bayes, and decision trees, and comparing their performance. To evaluate the effectiveness of each algorithm, we will display the confusion matrix on the test set. We will also create visualizations using a separate function, which can be called to obtain all necessary scores and visualizations for comparison.

Design
The PyML package consists of several modules, each containing the implementation of a specific machine learning algorithm. The module has classes for each of the models. Each class will have methods for training the model, making predictions, and evaluating the model's performance. We will also add a visualize function which will display the performance of all the three models at once which will make it easier for the user to decide on which will be his final model.
The project has also a module to perform the preprocessing steps like loading the data, performing normalization, and splitting the data into k-folds. Finally, we have a module called pyML which can be imported and run by anyone. pyML would run Naive Bayes, Decision Tree and Logistic Regression and display the results of all these algorithms so that the user can decide which algorithm to choose.
The package contains the following modules:

•	util_preprocess.py: This module contains functions for loading and preprocessing the spam data. It also contains functions for splitting the data into k-folds and plotting ROC curves.

•	logistic_regression.py: This module contains the implementation of logistic regression using gradient descent. It contains a function called logistic_regression that performs k-fold cross-validation on the data and trains the model on each fold.

•	naives_bayes.py: This module contains the implementation of the bernoulli naive bayes algorithm. It has a function called train which trains the naives bayes classifier on the data by splitting it into k-folds. This module also contains a function to perform predictions on the test set.

•	decision_tree.py: This module implements the decision tree algorithm for classification. It defines a Node class with methods to create and predict with a decision tree. The create_tree_util() function creates the decision tree recursively, and the split_data() function splits the data into left and right subtrees based on a splitting feature and threshold value. The get_entropy() function calculates the entropy of a given set of labels, and the get_feature_to_split() function returns the splitting feature with the highest information gain. Finally, the predict_label() method predicts the label of a given input feature vector by traversing the decision tree.

•	pyML.py: This module can be imported by any user to train all the above models and visualize the results.


Following are a few reasons why a module like pyML can be useful to the users:

•	Timesaving: Training a single ML model can be a time-consuming process, and it often requires several rounds of trial and error to find the best model. By training multiple models at once, this module can save a significant amount of time and effort.

•	Comparison of Models: By training and visualizing the results of multiple models, users can easily compare the performance of different models and identify the best one for their specific use case. This can help to improve the accuracy of predictions and decision-making.

•	Ease of Use: For users who are not experts in ML, this module can simplify the process of building and testing models. Rather than needing to manually train and evaluate each model, they can use this module to streamline the process and quickly generate results.

•	Flexibility: Different ML algorithms can be better suited for different types of data and problems. This module provides flexibility to try out multiple algorithms and identify the best one for a given problem.

•	Insights: The visualizations generated by this module, such as ROC curves, can provide insights into the performance of each model, helping users to identify strengths and weaknesses and make improvements.

***

# Execution

The following are the commands to use to execute the project:

1) Make sure that python has been installed on your computer and then navigate to the pyML folder and place all the files in the folder.
2) Install all the requirements using the "pip install -r requirements.txt" command 
3) Then execute the python file using the command "python -m PyML.pyML". This command will train and build the three models for the spambase.data file which has been provided. To run these models on a different database we can add the data in the code and add the file as well.
4) To run the test cases python file we need to run the command "python -m PyML.Unittest_cases"

Note: While executing the commands ROC curve graphs for the three models pop-up, close the visualisations to continue execution.
