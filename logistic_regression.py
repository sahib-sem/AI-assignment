import tensorflow as tf
import math
import numpy as np  
class LogisticRegression:
    
    def __init__(self, learning_rate=1, num_iterations=100):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.train_images = self.train_images.reshape((-1, 784))
        self.test_images = self.test_images.reshape((-1, 784))
        self.train_images = self.normalize(self.train_images)
        self.test_images  = self.normalize(self.test_images)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.train(self.train_images,self.train_labels)
       
    def normalize(self, image_list):
        normalized_images = (image_list / 255.0) 
        return normalized_images

    def softmax(self, x):
        max_value = max(x)
        exp_scores = [math.exp(score - max_value) for score in x]
        total_sum = sum(exp_scores)
        softmax_probs = [score / total_sum for score in exp_scores]
        return softmax_probs
    def dot_product(self,image, weight):
        image_arr = np.array(image)
        weight_arr = np.array(weight)
        return np.sum(image_arr * weight_arr)
    
    def train(self, x, y):
        feature_size = len(x[0])
        #prepare the weights for the 10 digits
        self.weights = [[0] * feature_size for _ in range(10)]
        
        for _ in range(self.num_iterations):
            #calculate the dot product of the weights and the features
            
            for index,image in enumerate(x):
                dot_products = []
                for idx, digit_weight in enumerate(self.weights):
                    #calculate the dot product of the weights and the features
                    dot_products.append(self.dot_product(image,digit_weight))
                    
                #calculate the sigmoid of the dot product
                soft_max_score = self.softmax(dot_products)
                
                for idx, digit_weight in enumerate(self.weights):
                    #calculate the error
                    error = 1 if idx == y[index] else 0 - soft_max_score[idx]
                    #update the weights
                    for i in range(feature_size):
                        digit_weight[i] -= self.learning_rate * error * image[i]
            
    def predict(self, image):
        linear_regression = []
        for _, digit_weight in enumerate(self.weights):
            linear_regression.append(self.dot_product(image,digit_weight))            
          
        probabilities = self.softmax(linear_regression)  
        return probabilities.index(max(probabilities))
            
        
        
obj = LogisticRegression()
# let count how many tests we got right
count = 0
for index,test_image in enumerate(obj.test_images):
    if obj.predict(test_image) == obj.test_labels[index]:
       count += 1 
    else:
        pass
    
print("Accuracy: ",count/len(obj.test_images)*100,"%")