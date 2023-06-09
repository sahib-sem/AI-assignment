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
        
    def prepare(self):
        mean = []
        stdv = []
        for lis in zip(*self.train_images):
            mean.append(self.mean(lis))
            stdv.append(self.standard_div(lis))
            
        return mean,stdv
        
            
    def data_statistics(self):
        for key in self.class_group:
            mean,stdv = self.prepare()
            #let's calculate mean and standard deviation of each class
            self.class_stat[key].append(mean)
            self.class_stat[key].append(stdv)

            
    def probability(self,x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    def class_probability(self,data):
        #let's get the mean representation of our incoming data
      
        class_prob = defaultdict(int)
        #let's calculate the probability of each class
        for key in self.class_stat:
            for index,value in enumerate(data):
                class_prob[key] *= self.probability(value,self.class_stat[key][0][index],self.class_stat[key][1][index])
            
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

        
        
        

