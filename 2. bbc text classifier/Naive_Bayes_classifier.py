from collections import defaultdict
from math import exp, sqrt, pi
import math
import random
import time

from matplotlib import pyplot as plt
from tabulate import tabulate

from utilities import read_file,  vectorize


def mean(data):
    return sum(data) / len(data)

def calculate_probability(x, mean, stdev, k ):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 + k )))
    return (1 / (sqrt(2 * pi) * (stdev + k))) * exponent

def stdev(data):
    avg = mean(data)
    process_data = [(x- avg) ** 2 for x in data]

    return sum(process_data)/ (len(process_data) - 1 )

class BayesClassifier:

    def __init__(self) -> None:
        self.bag_of_words = vectorize()
        self.label = self.load_label()
        self.td_idf = vectorize(idf_tf=True)
        self.term_presence = vectorize(term_presence=True)
        
    def dataset_load_bag_of_words(self):
        dataset = self.text_with_label(self.label, self.bag_of_words)
        return dataset
    
    def dataset_load_td_idf(self):
        dataset = self.text_with_label(self.label, self.td_idf)
        return dataset
    def dataset_load_term_presence(self):
        dataset = self.text_with_label(self.label, self.term_presence)
        return dataset
    
    def load_label(self):
        label = {int(line.split()[0]):int(line.split()[-1]) for line in read_file('bbc/bbc.classes')[4:]}
        return label
    
    def text_with_label(self, label_class, vector):
        dataset = []

        for text_no, text in vector.items():
            dataset.append((label_class[text_no - 1], text))
        
        
        random.shuffle(dataset)

        return dataset
    
    def separate_dataset(self, dataset):
        trainining_set = []
        test_set = []

        pivot = int(0.8 * len(dataset))

        for idx in range(len(dataset)):
            if idx < pivot:
                trainining_set.append(dataset[idx])
            else:
                test_set.append(dataset[idx])
        
        return trainining_set, test_set

    def summerize_data(self, dataset):
        summary_data = []
        for column in zip(*dataset):
            column = [float(c) for c in column]
            avg = mean(column)
            sd = stdev(column)
            summary_data.append((avg, sd, len(column)))
        
        return summary_data

    
    def separate_by_class(self, dataset):
        separated = defaultdict(list)

        for data in dataset:

            label = data[0]
            separated[label].append( data[1])
        
        return separated
    def summerize_data_by_class(self, dataset):
        separeted = self.separate_by_class(dataset)
        summeries = {}

        for label, data in separeted.items():
            summeries[label] = self.summerize_data(data)
        
        return summeries 
    
    def bayes_classifier(self,summerized_data,  test_data, total_data, smothing_factor):
        
        class_probability = {}

        for label, feature_summary in summerized_data.items():
            class_probability[label] = math.log(len(feature_summary) / total_data)

            for i in range(len(feature_summary)):

                mean, stdev, count = feature_summary[i]

                if test_data[i] == 0:
                    continue
            
                prop = calculate_probability(test_data[i], mean, stdev, smothing_factor)
                if prop == 0:
                    continue
                class_probability[label] += math.log(prop)
        
        return max(class_probability, key=class_probability.get)

    def evaluate_algorithm(self,summerized_data, test_set, total, smothing_factor):
        correct = 0

        for data in test_set:
            label = data[0]
            test_data = data[1]
            predicted = self.bayes_classifier(summerized_data, test_data, total,smothing_factor)
            if label == predicted:
                correct += 1
        return correct / len(test_set)
    



clf = BayesClassifier()

datasets = [clf.dataset_load_bag_of_words, clf.dataset_load_td_idf, clf.dataset_load_term_presence]
method_of_feature_extraction = ['Bag_of_Words_Naive_bayes ', 'TD-IDF_Naive_bayes', 'Term_Presence_Naive_bayes']

table_data = []

for idx, dataset_load in enumerate(datasets):
    analysis = defaultdict(int)
    for _ in range(5):

        dataset = dataset_load()
        training_set, test_set = clf.separate_dataset(dataset)
        summerized_data = clf.summerize_data_by_class(training_set)
        total = len(training_set)

        smoothing_rates = [0.1, 0.5, 1.0, 10, 100]
        
        for k in smoothing_rates:
            percent = clf.evaluate_algorithm(summerized_data, test_set, len(training_set), k) * 100
            
            analysis[k] += percent
    
    analysis = dict(sorted(analysis.items(), key=lambda x: x[0]))
    accuracies = [val /5 for val in analysis.values()]
    table_data.append(accuracies)

    plt.plot(smoothing_rates, accuracies, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Learning Rate')
    plt.grid(True)
    plt.savefig(f'{method_of_feature_extraction[idx]}.png')
    plt.close()

table_data.insert(0, smoothing_rates)
table_data_formated = [data for data in zip(*table_data)]
headers = ["Smoothing Factor", "Bag of Words", "TF-IDF", "Term Presence"]

table = tabulate(table_data_formated, headers, tablefmt="fancy_grid")

print(table)
