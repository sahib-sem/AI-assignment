import tensorflow as tf # only used to access the mnist dataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class NaiveBayes:
    
    def __init__(self, smoothing_factor):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.class_group = defaultdict(list)
        self.train_images = self.train_images.reshape((-1, 784))
        self.test_images = self.test_images.reshape((-1, 784))
        self.train_images = self.normalize(self.train_images)
        self.test_images = self.normalize(self.test_images)
        self.class_stat = defaultdict(list)
        self.smoothing_factor = smoothing_factor

    def divide_by_class(self):
        for index, image in enumerate(self.train_images):
            image_class = self.train_labels[index]
            self.class_group[image_class].append(image)
        self.data_statistics()

    def normalize(self, image_list):
        normalized_images = (image_list / 255.0) ** 2
        return normalized_images

    def prepare(self, group_list):
        mean = []
        stdv = []
        for lis in zip(*group_list):
            mean.append(self.mean(lis))
            stdv.append(self.standard_div(lis))
        return mean, stdv

    def data_statistics(self):
        for key in self.class_group:
            mean, stdv = self.prepare(self.class_group[key])
            self.class_stat[key].append(mean)
            self.class_stat[key].append(stdv)

    def probability(self, x, mean, stdev):
        exponent = -((x - mean) ** 2) / (2 * stdev ** 2 + self.smoothing_factor)
        denominator = np.sqrt(2 * np.pi * stdev ** 2 + self.smoothing_factor)
        return np.exp(exponent) / denominator

    def class_probability(self, data):
        class_prob = defaultdict(float)
        for key in self.class_stat:
            class_prob[key] += np.log(len(self.class_group[key]) / len(self.train_images))
            for index, value in enumerate(data):
                if value == 0:
                    continue
                conditional_probability = self.probability(value, self.class_stat[key][0][index], self.class_stat[key][1][index])
    
                class_prob[key] += np.log(conditional_probability)
        return class_prob

    def predict(self, data):
        class_prob = self.class_probability(data)
        return max(class_prob, key=class_prob.get)
        
    def mean(self, lis):
        return sum(lis) / float(len(lis))

    def standard_div(self, lis):
        mean = self.mean(lis)
        variance = sum([(x - mean) ** 2 for x in lis]) / float(len(lis) - 1)
        return variance ** 0.5

    def apply_pca(self, num_components):
        covariance_matrix = np.cov(self.train_images.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        top_indices = np.argsort(eigenvalues)[::-1][:num_components]
        top_eigenvectors = eigenvectors[:, top_indices]
        
        self.train_images = np.dot(self.train_images, top_eigenvectors)
        self.test_images = np.dot(self.test_images, top_eigenvectors)
        self.divide_by_class()

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
        self.divide_by_class()

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

def general_feature_extraction(obj):
    obj.divide_by_class()
    
def principal_component_analysis(obj):
    obj.apply_pca(50)
    
    
smoothing_factors = [0.1,0.5,1.0,10,100]
accuracies = []
feature_extraction = [principal_component_analysis]

for feature in feature_extraction:
        accuracy = []
        for smoothing_factor in smoothing_factors:
            obj = NaiveBayes(smoothing_factor)
            feature(obj)
            count = 0
            for index, test_image in enumerate(obj.test_images):
                if obj.predict(test_image) == obj.test_labels[index]:
                    count += 1
            accuracy.append(count / len(obj.test_images) * 100)
            print("Accuracy:", count / len(obj.test_images) * 100, "%")
            
        accuracies.append(accuracy)
        
for accuracy in accuracies:
    plt.plot(smoothing_factors, accuracy, marker='o')
    plt.xlabel('smoothing factors')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. smoothing factors')
    plt.grid(True)
    plt.show()
