from collections import defaultdict
from math import exp, sqrt, log,pi
import matplotlib.pyplot as plt
from heapq import *
import weather_data

class NaiveBayesClassifier:
    def __init__(self, smoothing=0.00000000001):
        self.smoothing = smoothing

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance) + self.smoothing

    def summarize_dataset(self, dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        return summaries
    
    def normalize(self,dataset):
        for key , val in dataset.items():
            normalized = []
            for pixel_arr in val:
                    normalized_pixel = []
                    for pixel in pixel_arr:
                        normalized_pixel.append(pixel) 
                    normalized.append(normalized_pixel)

            dataset[key] = normalized

        return dataset
    
    def summarize_by_class(self, dataset):
        dataset = self.normalize(dataset)
        summaries = defaultdict(list)
        for class_value, rows in dataset.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return self.smoothing + (1 / (sqrt(2 * pi) * stdev)) * exponent

    def calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = defaultdict(lambda: 1.0)
        for class_value, class_summaries in summaries.items():
            for i in range(len(class_summaries)):
                    mean, stdev, count = class_summaries[i]
                    if mean != 0 and row[i] != 0 and stdev != 0:
                        probabilities[class_value] += log(self.calculate_probability(row[i], mean, stdev))
                    else:
                        continue
            probabilities[class_value] += log(summaries[class_value][0][2] / float(total_rows))
        return probabilities


        


    


dataset = weather_data.read_csv_file("weather_data\weather.csv")
test_set = weather_data.read_csv_file("weather_data\weather_test.csv")



smoothing_factors = [0.1,0.5,1.0,10,100]
accuracy = []

for smoothing_factor in smoothing_factors:
    classify = NaiveBayesClassifier(smoothing_factor)
    summaries = classify.summarize_by_class(dataset)
    test_summaries = classify.summarize_by_class(test_set)
    count = 0
    for lable, images in test_set.items():
        for image in images:
            predictions = [(-value,label) for label,value in classify.calculate_class_probabilities(summaries, image).items()]
            heapify(predictions)
            predicted = heappop(predictions)
            if lable == predicted[1]:
                count += 1


    total_rows = sum([test_summaries[label][0][2] for label in test_summaries])
    accuracy.append((count / total_rows) * 100)
    print("Accuracy" , count / total_rows)
    
plt.plot(smoothing_factors, accuracy, marker='o')
plt.xlabel('Smoothing factor')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Smoothing factor')
plt.grid(True)
plt.show()


