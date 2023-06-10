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
        # self.train(self.train_images, self.train_labels)

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
    def apply_pca(self, num_components):
        covariance_matrix = np.cov(self.train_images.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        top_indices = np.argsort(eigenvalues)[::-1][:num_components]
        top_eigenvectors = eigenvectors[:, top_indices]
        
        self.train_images = np.dot(self.train_images, top_eigenvectors)
        self.test_images = np.dot(self.test_images, top_eigenvectors)
        self.train(self.train_images, self.train_labels)
        
    def apply_lbp(self):
        lbp_images = []
        for image in self.train_images:
            lbp_image = self.local_binary_pattern(image.reshape((28, 28)), 3, 8)
            lbp_images.append(lbp_image.ravel())
        self.train_images = np.array(lbp_images)

        lbp_images = []
        for image in self.test_images:
            lbp_image = self.local_binary_pattern(image.reshape((28, 28)), 3, 8)
            lbp_images.append(lbp_image.ravel())
        self.test_images = np.array(lbp_images)
        self.train(self.train_images, self.train_labels)

    def local_binary_pattern(self, image, radius, num_points):
        rows, cols = image.shape
        lbp_image = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center_pixel = image[i, j]
                binary_code = 0
                for k in range(num_points):
                    angle = 2 * np.pi * k / num_points
                    x = i + int(round(radius * np.cos(angle)))
                    y = j + int(round(radius * np.sin(angle)))
                    neighbor_pixel = image[x, y]
                    if neighbor_pixel >= center_pixel:
                        binary_code += 2 ** k
                lbp_image[i, j] = binary_code
        return lbp_image

    def predict(self, image):
        dot_products = np.dot(self.weights, image)
        probabilities = self.softmax(dot_products)
        return np.argmax(probabilities)




def general_feature_extraction(obj):
    obj.train(obj.train_images, obj.train_labels)
    
def principal_component_analysis(obj):
    obj.apply_pca(50)
    


learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracies = []
feature_extraction = [general_feature_extraction,principal_component_analysis]

for feature in feature_extraction:
        accuracy = []
        for learning_rate in learning_rates:
            obj = LogisticRegression(learning_rate)
            feature(obj)
            count = 0
            for index, test_image in enumerate(obj.test_images):
                if obj.predict(test_image) == obj.test_labels[index]:
                    count += 1
            accuracy.append(count / len(obj.test_images) * 100)
            print("Accuracy:", count / len(obj.test_images) * 100, "%")
            
        accuracies.append(accuracy)
        
for accuracy in accuracies:
    plt.plot(learning_rates, accuracy, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Learning Rate')
    plt.grid(True)
    plt.show()