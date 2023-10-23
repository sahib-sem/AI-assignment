
from tabulate import tabulate
from matplotlib import pyplot as plt
import numpy as np
from utilities import generate_regression_data

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=10):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None

    def softmax(self, x):
        max_value = np.max(x)
        exp_scores = np.exp(x - max_value)
        softmax_probs = exp_scores / np.sum(exp_scores)
        return softmax_probs

    def dot_product(self, vector, weight):
        return np.dot(vector, weight)

    def train(self, x, y):
        feature_size = len(x[0])
        self.weights = np.zeros((5, feature_size))

        for _ in range(self.num_iterations):
            for index, vector in enumerate(x):
                dot_products = np.dot(self.weights, vector)
                soft_max_score = self.softmax(dot_products)

                targets = np.zeros(5)
                targets[y[index]] = 1

                errors = targets - soft_max_score
                weight_updates = self.learning_rate * np.outer(errors, vector)
                self.weights += weight_updates

    def predict(self, vector):
        dot_products = np.dot(self.weights, vector)
        probabilities = self.softmax(dot_products)
        return np.argmax(probabilities)





term_presence_training_data , term_presence_training_label, term_presence_test_data, term_presence_test_label = generate_regression_data(term_presence_passed = True)
TD_IDF_training_data , TD_IDF_training_label, TD_IDF_test_data, TD_IDF_test_label = generate_regression_data(tf_idf_passed= True)
bag_of_words_training_data , bag_of_words_training_label, bag_of_words_test_data, bag_of_words_test_label = generate_regression_data()
method_of_feature_extraction = ['Bag_of_Words_logistic_regression', 'TD-IDF_logistic_regression', 'Term_Presence_logistic_regression']

datasets = [(bag_of_words_training_data, bag_of_words_training_label, bag_of_words_test_data, bag_of_words_test_label), 
            (TD_IDF_training_data, TD_IDF_training_label, TD_IDF_test_data, TD_IDF_test_label),
            (term_presence_training_data, term_presence_training_label, term_presence_test_data, term_presence_test_label),
            
            ]

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]

table_data = []

for idx, dataset in enumerate(datasets):

    accuracies = []

    for lr in learning_rates:
        x_train, x_label, y_test, y_label = dataset
        count = 0
        clf = LogisticRegression(learning_rate=lr)
        clf.train(np.array(x_train), np.array(x_label))
        for vector , label in zip(y_test, y_label):
            if clf.predict(vector) == label:
                count += 1
        accuracies.append(count / len(y_label) * 100)
    
    table_data.append(accuracies)
    


    plt.plot(learning_rates, accuracies, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Learning Rate')
    plt.grid(True)
    plt.savefig(f'{method_of_feature_extraction[idx]}.png')
    plt.close()

table_data.insert(0, learning_rates)
table_data_formated = [data for data in zip(*table_data)]
headers = ["Learning Rate", "Bag of Words", "TF-IDF", "Term Presence"]

table = tabulate(table_data_formated, headers, tablefmt="fancy_grid")

print(table)