from cmath import exp, pi, sqrt
from collections import defaultdict
import math
import random


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

data_mtx = read_file('bbc/bbc.mtx')[2:]
def length_of_document(data):
    
    document_length = defaultdict(int)
    for line in data:
        word_no, text_no, freq = [int(float(x)) for x in line.split()]
        document_length[text_no] += freq
    
    return document_length

def count_words(data):
    word_count = defaultdict(lambda: defaultdict(int))
    for line in data:
        word_no, text_no, freq = [int(float(x)) for x in line.split()]
        word_count[word_no][text_no] += freq
    
    return word_count

def Tf(word_count, document_length):
    tf = defaultdict(lambda: defaultdict(int))
    for word, text in word_count.items():
        for text_no, count in text.items():
            tf[word][text_no] = count / document_length[text_no]
    
    return tf

def Idf(word_count, document_length):
    idf = defaultdict(int)
    for word, text in word_count.items():
        idf[word] = math.log(len(document_length) / len(text))
    
    return idf

def vectorize(idf_tf = False, term_presence = False):
    chunk_data = read_file('bbc/bbc.mtx')[2:]

    if idf_tf:
        document_length = length_of_document(chunk_data)
        word_count = count_words(chunk_data)
        tf = Tf(word_count, document_length)
        idf = Idf(word_count, document_length)
        text_word_map = {text_no:[0 for i in range(9635)] for text_no in range(1, 2226)}
        for word, text in tf.items():
            for text_no, freq in text.items():
                text_word_map[text_no][word - 1] = freq * idf[word]
        return text_word_map
    
    text_word_map = {text_no:[0 for i in range(9635)] for text_no in range(1, 2226)}
    
    for line in chunk_data:

        word_no, text_no , freq = [int(float(x)) for x in line.split()]
        if term_presence:
            text_word_map[text_no][int(word_no) - 1] = 1
        else:
            text_word_map[text_no][int(word_no) - 1] += freq

    return text_word_map

def separate_dataset(dataset):
    trainining_set = []
    test_set = []

    pivot = int(0.8 * len(dataset))

    for idx in range(len(dataset)):
        if idx < pivot:
            trainining_set.append(dataset[idx])
        else:
            test_set.append(dataset[idx])
    
    return trainining_set, test_set

def text_with_label(label_class, vector):
    dataset = []

    for text_no, text in vector.items():
        dataset.append((label_class[text_no - 1], text))
    
    
    random.shuffle(dataset)

    return dataset



def generate_regression_data(tf_idf_passed = False, term_presence_passed = False):

    label = {int(line.split()[0]):int(line.split()[-1]) for line in read_file('bbc/bbc.classes')[4:]}

    if term_presence_passed:
        vector = vectorize(idf_tf = tf_idf_passed)
    elif term_presence_passed:
        vector = vectorize(term_presence = term_presence_passed)
    else:
        vector = vectorize()

    dataset = text_with_label(label, vector)
    trainining_set, test_set = separate_dataset(dataset)
    x_training_data, x_label , y_training_data, y_label = [], [], [], []
    
    for data in trainining_set:
        x_training_data.append(data[1])
        x_label.append(data[0])
    
    for data in test_set:
        y_training_data.append(data[1])
        y_label.append(data[0])

    return x_training_data, x_label , y_training_data, y_label


