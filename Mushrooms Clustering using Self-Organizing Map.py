import csv, tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

height = 8
width = 8
input_dimension = 3

def load_data(filename):
    raw_dataset = []
    with open(filename) as f:
        next(csv.reader(f))
        for row in csv.reader(f):       
            features_selection = [row[0]] + [row[9]] + [row[17]] + [f for f in row[20:]]
            raw_dataset.append(features_selection)
    return raw_dataset

def preprocess(dataset):
    new_dataset = []
    for xFeatures in dataset:
        xFeatures[0] = convert_cap_shape(xFeatures[0])
        xFeatures[1] = convert_stalk_shape(xFeatures[1])
        xFeatures[2] = convert_ring_number(xFeatures[2])
        xFeatures[3] = convert_population(xFeatures[3])
        xFeatures[4] = convert_habitat(xFeatures[4])
        new_dataset.append(xFeatures)
    return new_dataset

def convert_cap_shape(cap_shape):
    if cap_shape == "b": return 0
    elif cap_shape == "c": return 1
    elif cap_shape == "x": return 2
    elif cap_shape == "f": return 3
    elif cap_shape == "k": return 4
    else: return 5

def convert_stalk_shape(stalk_shape):
    if stalk_shape == "e": return 0
    else: return 1

def convert_ring_number(ring_number):
    if ring_number == "n": return 0
    elif ring_number == "o": return 1
    else: return 2

def convert_population(population):
    if population == "a": return 0
    elif population == "c": return 1
    elif population == "n": return 2
    elif population == "s": return 3
    elif population == "v": return 4
    else: return 5

def convert_habitat(habitat):
    if habitat == "g": return 0
    elif habitat == "l": return 1
    elif habitat == "m": return 2
    elif habitat == "p": return 3
    elif habitat == "u": return 4
    elif habitat == "w": return 5
    else: return 6

def normalize(dataset):
    features = np.array([data for data in dataset])
    maxF = features.max(axis=0)
    minF = features.min(axis=0)

    new_dataset = []
    for features in dataset:
        for i in range(len(features)):
            features[i] = (features[i] - minF[i]) / (maxF[i] - minF[i]) * (1 - 0) + 0
        new_dataset.append(features)
    return new_dataset

def apply_pca(features):
    pca  = PCA(n_components=3)
    new_features = pca.fit_transform(features)
    return new_features

class SOM:
    def __init__(self, height, width, input_dimension):
        self.height = height
        self.width = width
        self.input_dimension = input_dimension

        self.nodes = [tf.to_float([x, y]) for y in range(height) for x in range(width)]
        self.weight = tf.Variable(tf.random_normal([self.width * self.height, input_dimension]))
        
        self.x = tf.placeholder(tf.float32, [input_dimension])
        self.bmu = self.get_bmu(self.x) # Best Matching Unit (the winning node) # (1) COMPETITION process
        self.update = self.update_neighbors(self.bmu, self.x) # (2) COOPERATION and (3) SYPNATIC ADAPTATION process in one function

    def get_bmu(self, x): # (1) COMPETITION process
        square_diff = tf.square(x - self.weight)
        distance = tf.sqrt(tf.reduce_sum(square_diff, 1)) # axis >> 0 = vertical sum || 1 = horizontal sum
        bmu_index = tf.argmin(distance, 0) # argmin returns index
        bmu_node = tf.to_float([tf.mod(bmu_index, self.width), tf.div(bmu_index, self.height)]) # node coordinate [x, y]
        return bmu_node
    
    def update_neighbors(self, bmu, x): # (2) COOPERATION and (3) SYPNATIC ADAPTATION process
        # Calculate NS (Neighbors Strength)
        sigma = tf.to_float(tf.maximum(self.width, self.height) / 2)
        square_diff = tf.square(bmu - self.nodes)
        distance = tf.sqrt(tf.reduce_sum(square_diff, 1)) # axis >> 0 = vertical sum || 1 = horizontal sum
        ns = tf.exp(tf.negative(tf.div(tf.square(distance), 2 * tf.square(sigma))))
        
        learning_rate = .1
        ns_learning_rate = tf.multiply(ns, learning_rate)
        
        # tile for multiple, e.g. [1] into [1, 1, 1]
        # slice for slicing array
        ns_learning_rate_stacked = tf.stack([tf.tile(tf.slice(ns_learning_rate, [i], [1]), [self.input_dimension]) for i in range(self.width * self.height)])
        x_w_diff = tf.subtract(x, self.weight)      
        weight_diff = tf.multiply(ns_learning_rate_stacked, x_w_diff)

        new_weight = tf.add(self.weight, weight_diff)
        return tf.assign(self.weight, new_weight) # self.weight += new_weight

    def train_SOM(self, dataset, number_of_epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for a in range(1, number_of_epoch+1):
                for data in dataset:
                    feed = {self.x: data}
                    sess.run(self.update, feed)
            
            self.cluster = [[] for a in range(self.width)]
            self.nodes_val = sess.run(self.nodes)
            self.weight_val = sess.run(self.weight)

            for b, node in enumerate(self.nodes_val):
                self.cluster[int(node[0])].append(self.weight_val[b])

######################## LOAD DATA #######################
raw_dataset = load_data("O202-COMP7117-KK02-00-clustering.csv")
new_dataset = preprocess(raw_dataset)
new_dataset = normalize(new_dataset)
new_dataset = apply_pca(new_dataset)

###################### SOM PROCESS #######################
som = SOM(height, width, input_dimension)
# [WARNING] This process will take a very long time due to epoch = 5000
# For efficient result visualization, change epoch to the lower number possible
som.train_SOM(new_dataset, 5000)

################## RESULT VISUALIZATION ##################
plt.figure("Bee Research Center - SOM Result")
plt.imshow(som.cluster)
plt.show()