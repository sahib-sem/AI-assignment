import math
import tensorflow as tf

class NaiveBayes:
    def fit(self, X, y):
        self.classes = list(set(y))
        self.num_classes = len(self.classes)
        self.num_features = len(X[0])

        self.priors = [0] * self.num_classes
        self.means = [[0] * self.num_features for _ in range(self.num_classes)]
        self.variances = [[0] * self.num_features for _ in range(self.num_classes)]

        for idx, c in enumerate(self.classes):
            X_c = [X[i] for i in range(len(X)) if y[i] == c]

            self.priors[idx] = len(X_c) / len(X)
            self.means[idx] = [sum(feature) / len(X_c) for feature in zip(*X_c)]

            for i in range(self.num_features):
                variance = sum((x[i] - self.means[idx][i]) ** 2 for x in X_c) / len(X_c)
                self.variances[idx][i] = variance
                
    def prepare(self, image_list):
        # Preprocess the image data, e.g., normalize pixel values
        normalized_images = image_list / 255.0
        return normalized_images
        
    def predict(self, X):
        preds = []

        
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = math.log(self.priors[idx])
            likelihood = sum(math.log(self.calculate_probability(self.means[idx][i], self.variances[idx][i], X[i])) for i in range(self.num_features))
            posterior = prior + likelihood
            posteriors.append(posterior)

        preds.append(self.classes[posteriors.index(max(posteriors))])

        return preds[0]

    def calculate_probability(self, mean, variance, x):
        eps = 1e-9  # Small value to avoid math domain error
        exponent = -((x - mean) ** 2) / (2 * variance ** 2 + eps)
        denominator = math.sqrt(2 * math.pi * variance ** 2 + eps)
        return math.exp(exponent) / denominator

obj = NaiveBayes()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

obj.fit(train_images, train_labels)
count = 0
for index, test_image in enumerate(test_images):
    normalized_image = obj.prepare(test_image)
    if obj.predict(normalized_image) == test_labels[index]:
        count += 1 

print("Accuracy: ", count / len(obj.test_images) * 100, "%")
