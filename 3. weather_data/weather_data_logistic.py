import tensorflow as tf
import numpy as np
import math
import weather_data
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, train_images,train_labels,learning_rate=0.01, num_iterations=100):
        
        self.train_images = train_images
        self.train_labels = train_labels
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.train(self.train_images, self.train_labels)

    def softmax(self, x):
        max_value = np.max(x)
        exp_scores = np.exp(x - max_value)
        softmax_probs = exp_scores / np.sum(exp_scores)
        return softmax_probs

    def dot_product(self, image, weight):
        return np.dot(image, weight)

    def train(self, x, y):
        feature_size = len(x[0])
        self.weights = np.zeros((2, feature_size))

        for _ in range(self.num_iterations):
            for index, image in enumerate(x):
                dot_products = np.dot(self.weights, image)
                soft_max_score = self.softmax(dot_products)

                targets = np.zeros(2)
                targets[int(y[index])] = 1

                errors = targets - soft_max_score
                weight_updates = self.learning_rate * np.outer(errors, image)
                self.weights += weight_updates

    def predict(self, image):
        dot_products = np.dot(self.weights, image)
        probabilities = self.softmax(dot_products)
        return np.argmax(probabilities)


data_set = weather_data.read_csv_file("weather_data\weather.csv")
test_set = weather_data.read_csv_file("weather_data\weather_test.csv")

def organizer(data_set):
    Labels = []
    Data = []
    for label,data in data_set.items():
        for lis in data:
            Data.append(lis)
            Labels.append(label)
        
    return (Labels,Data)
        

train_label,train_data = organizer(data_set)
test_label,test_data = organizer(test_set)
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracy = []
for learning_rate in learning_rates:
    obj = LogisticRegression(train_data,train_label,learning_rate)
    count = 0
    for index, test in enumerate(test_data):   
        if obj.predict(test) == test_label[index]:
            count += 1
    accuracy.append(count / len(test_data)* 100)
    print("Accuracy:", count / len(test_data)* 100, "%")

plt.plot(learning_rates, accuracy, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Learning Rate')
plt.grid(True)
plt.show()