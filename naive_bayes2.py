import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from math import *
class NaiveBayes:
    
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.class_group = defaultdict(list)
        self.train_images = self.train_images.reshape((-1, 784))
        self.test_images = self.test_images.reshape((-1, 784))
        self.train_images = self.normalize(self.train_images)
        self.test_images  = self.normalize(self.test_images)
        self.class_stat = defaultdict(list)
        self.divide_by_class()
        self.smoothing_factor = 0.1

        
    def divide_by_class(self):
        for index,image in enumerate(self.train_images):
            image_class = self.train_labels[index]
            self.class_group[image_class].append(image)
        self.data_statistics()
        
    def normalize(self, image_list):
        # Preprocess the image data, e.g., normalize pixel values
        normalized_images = image_list / 255.0
        return normalized_images
        
    def prepare(self,group_list):
        mean = []
        stdv = []
        for lis in zip(*group_list):
            mean.append(self.mean(lis))
            stdv.append(self.standard_div(lis))
            
        return mean,stdv
        
            
    def data_statistics(self):
        for key in self.class_group:
            mean,stdv = self.prepare(self.class_group[key])
            #let's calculate mean and standard deviation of each class
            self.class_stat[key].append(mean)
            self.class_stat[key].append(stdv)

            
    def probability(self,x, mean, stdev):
        smoothing_factor = 0.1
        exponent = -((x - mean) ** 2) / (2 * stdev ** 2 + smoothing_factor)
        denominator = math.sqrt(2 * math.pi * stdev ** 2 + smoothing_factor)
        return math.exp(exponent) / denominator
    def class_probability(self,data):
        #let's get the mean representation of our incoming data
      
        class_prob = defaultdict(float)
        #let's calculate the probability of each class
        for key in self.class_stat:
            #probability of the class
            class_prob[key] = math.log(len(self.class_group[key])/len(self.train_images))
            for index,value in enumerate(data):
                if value == 0:
                    continue
                class_prob[key] += math.log(self.smoothing_factor + self.probability(value,self.class_stat[key][0][index],self.class_stat[key][1][index]))
            
        return class_prob
    
    def predict(self,data):
        class_prob = self.class_probability(data)
        #let's get the class with the highest probability
        return max(class_prob, key=class_prob.get)
        
        
    def mean(self,lis):
        return sum(lis)/float(len(lis))
    def standard_div(self,lis):
        mean = self.mean(lis)
        variance = sum([(x-mean)**2 for x in lis])/float(len(lis)-1)
        return variance**0.5
        
        
    
    
obj = NaiveBayes()
# let count how many tests we got right
count = 0
for index,test_image in enumerate(obj.test_images):
    if obj.predict(test_image) == obj.test_labels[index]:
       count += 1 
    else:
        pass
    
print("Accuracy: ",count/len(obj.test_images)*100,"%")

        
        
        

