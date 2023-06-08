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
        self.class_stat = defaultdict(list)
        self.divide_by_class()

        
    def divide_by_class(self):
        for index,image in enumerate(self.train_images):
            image_class = self.train_labels[index]
            self.class_group[image_class].append(image)
        self.data_statistics()
        
    def prepare(image_list):
        pass
        
            
    def data_statistics(self):
        for key in self.class_group:
            flat_list = [item for sublist in self.class_group[key] for item in sublist]
            #let's calculate mean and standard deviation of each class
            self.class_stat[key].append(self.mean(flat_list))
            self.class_stat[key].append(self.standard_div(flat_list))

            
    def probability(self,x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    def class_probability(self,data):
        #let's get the mean representation of our incoming data
        stdv = self.standard_div(data)
        class_prob = defaultdict()
        #let's calculate the probability of each class
        for key in self.class_stat:
            class_prob[key] = (self.probability(stdv,self.class_stat[key][0],self.class_stat[key][1]))  
            
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

        
        
        

