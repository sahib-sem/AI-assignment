import tensorflow as tf
from collections import defaultdict
import math
import numpy as np

class NaiveBayes:
    
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.class_group = defaultdict(list)
        self.train_images = self.train_images.reshape((-1, 784))
        self.test_images = self.test_images.reshape((-1, 784))
        self.class_stat = defaultdict(list)
        self.divide_by_class()

    def divide_by_class(self):
        for index, image in enumerate(self.train_images):
            image_class = self.train_labels[index]
            self.class_group[image_class].append(image)
        self.data_statistics()

    def prepare(self, image_list):
        # Preprocess the image data, e.g., normalize pixel values
        normalized_images = image_list / 255.0
        return normalized_images
        
    def data_statistics(self):
        for key in self.class_group:
            flat_list = [item for sublist in self.class_group[key] for item in sublist]
            # Let's calculate mean and standard deviation of each class
            self.class_stat[key].append(self.mean(flat_list))
            self.class_stat[key].append(self.standard_div(flat_list))

    def probability(self, x, mean, stdev):
        stdev = max(stdev, 1e-9)  # Avoid division by zero
        exponent = -((x - mean) ** 2 / (2 * stdev ** 2))
        return math.log(1 / (math.sqrt(2 * math.pi) * stdev)) + exponent

    def class_probability(self, data):
        # Let's get the mean representation of our incoming data
        stdv = self.standard_div(data)
        class_prob = defaultdict(float)
        # Let's calculate the probability of each class
        for key in self.class_stat:
            class_prob[key] = self.probability(stdv, self.class_stat[key][0], self.class_stat[key][1])
            
        return class_prob
    
    def predict(self, data):
        class_prob = self.class_probability(data)
        
        # Let's get the class with the highest probability
        winner_class,biggest_prob = None,float('-inf')
        for key in class_prob:
            if class_prob[key] > biggest_prob:
                winner_class,biggest_prob = key,class_prob[key] 
                
        return winner_class 
        
    def mean(self, lis):
        return sum(lis) / float(len(lis))
    
    def standard_div(self, lis):
        mean = self.mean(lis)
        variance = sum([(x - mean) ** 2 for x in lis]) / float(len(lis) - 1)
        return math.sqrt(variance)
        
        
obj = NaiveBayes()

# Let's count how many tests we got right
count = 0
for index, test_image in enumerate(obj.test_images):
    normalized_image = obj.prepare(test_image)
    if obj.predict(normalized_image) == obj.test_labels[index]:
        count += 1 

print("Accuracy: ", count / len(obj.test_images) * 100, "%")
