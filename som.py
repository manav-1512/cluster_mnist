# Importing the necessary libraries
import numpy as np
import cv2
from imutils.paths import list_images
import matplotlib.pyplot as plt
import keras
from scipy.stats import mode

# Generating the Filepaths for all the images in the Test Set
imagePaths = sorted(list(list_images("testSet")))

# Preprocessing of the images
data = list()
tdata = list()
# Loop over the input images
for imagePath in imagePaths:
    # Loading image in grayscale format
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    image = image/255
    # # Loading the image in BGR format
    # cimage = cv2.imread(imagePath)
    # cimage = np.array(cimage)
    # cimage = cimage/255
    # Creating an array of images in BGR format to pass to pretrained CNN model
    tdata.append(image)
    # Flattening of the array of images to use in clustering
    im = list()
    for i in range(28):
        for j in range(28):
            im.append(image[i,j])
    data.append(im)

# Converting lists to numpy float arrays
tdata = np.array(tdata, dtype="float")
data = np.array(data, dtype="float")

# print(data.shape)

# # Feature Scaling and Extraction using Principal Component Analysis
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# data = pca.fit_transform(data)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y=1, input_len = 784, sigma = 0.45, learning_rate = 0.5, random_seed=0)
som.random_weights_init(data)
som.train_random(data = data, num_iteration = 100)

y = list()

for i, x in enumerate(data):
    w = som.winner(x)
    y.append(w[0])

y_kmeans = np.array(y)

# data = pca.inverse_transform(data)

# Loading the pretrained CNN model used to provide with labels
model = keras.models.load_model('mnist.model')

# data = pca.inverse_transform(data)

tdata = tdata.reshape(-1,28,28,1)

y_pre_pred = model.predict(tdata)

y_pred = []

for y in y_pre_pred:
    y_pred.append(np.where(y == np.amax(y)))

y_pred = np.array(y_pred)
y_pred = y_pred.reshape(-1)

y = np.zeros((10,10), dtype='int')

for i in range(10):
    yp = y_pred[np.where(y_kmeans == i)]
    for x in yp:
        y[i][x] +=1

acc = 100* np.sum(np.amax(y, axis = 1)) / y_kmeans.shape[0]


# Printing accuracy of KMenas Clustering
print('Accuracy = {:.2f} %'.format(acc))
