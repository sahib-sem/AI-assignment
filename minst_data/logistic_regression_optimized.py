import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.train_images = self.train_images.reshape((-1, 784))
        self.test_images = self.test_images.reshape((-1, 784))
        self.train_images = self.normalize(self.train_images)
        self.test_images = self.normalize(self.test_images)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.train(self.train_images, self.train_labels)

    def normalize(self, image_list):
        normalized_images = image_list / 255.0
        return normalized_images

    def softmax(self, x):
        max_value = np.max(x)
        exp_scores = np.exp(x - max_value)
        softmax_probs = exp_scores / np.sum(exp_scores)
        return softmax_probs

    def dot_product(self, image, weight):
        return np.dot(image, weight)

    def train(self, x, y):
        feature_size = len(x[0])
        self.weights = np.zeros((10, feature_size))

        for _ in range(self.num_iterations):
            for index, image in enumerate(x):
                dot_products = np.dot(self.weights, image)
                soft_max_score = self.softmax(dot_products)

                targets = np.zeros(10)
                targets[y[index]] = 1

                errors = targets - soft_max_score
                weight_updates = self.learning_rate * np.outer(errors, image)
                self.weights += weight_updates

    def predict(self, image):
        dot_products = np.dot(self.weights, image)
        probabilities = self.softmax(dot_products)
        return np.argmax(probabilities)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracy = []

for learning_rate in learning_rates:
    obj = LogisticRegression(learning_rate)
    count = 0
    for index, test_image in enumerate(obj.test_images):
        if obj.predict(test_image) == obj.test_labels[index]:
            count += 1
    accuracy.append(count / len(obj.test_images) * 100)
    print("Accuracy:", count / len(obj.test_images) * 100, "%")

plt.plot(learning_rates, accuracy, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Learning Rate')
plt.grid(True)
plt.show()