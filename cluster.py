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
    # Loading the image in BGR format
    cimage = cv2.imread(imagePath)
    cimage = np.array(cimage)
    cimage = cimage/255
    # Creating an array of images in BGR format to pass to pretrained CNN model
    tdata.append(cimage)
    # Flattening of the array of images to use in clustering
    im = list()
    for i in range(28):
        for j in range(28):
            im.append(image[i,j])
    data.append(im)

# Converting lists to numpy float arrays
tdata = np.array(tdata, dtype="float")
data = np.array(data, dtype="float")

# Feature Scaling and Extraction using Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
data = pca.fit_transform(data)

# Applying Kmeans Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 10, init='k-means++', n_init=1, random_state=0)
y_kmeans = kmeans.fit_predict(data)

# Loading the pretrained CNN model used to provide with labels
model = keras.models.load_model('mnist.model')

# Creating an array of labels using CNN model
y_pred = model.predict(tdata)
y = np.ones(y_pred.shape[0], dtype = 'int')
for i,x in enumerate(y_pred):
    y[i] = np.where(x == np.amax(x))[0]

# Getting Accuracy of Clustering
acc = 0.0
for i in range(10):
    m,c = mode(y[y_kmeans == i])
    c = c[0]
    acc += 100*c/(y[y_kmeans == i].shape[0])

# # Alternative method for getting accuracy
# acc = 0.0
# for x in range(10):
#     y_pred = model.predict(tdata[y_kmeans==x])
#     y = np.zeros(10, dtype = 'int')
#     for i in y_pred:
#         y[i == np.amax(i)] +=1
#     acc += 100*np.amax(y)/y_pred.shape[0]

# Printing accuracy of KMenas Clustering
print('Accuracy = {:.2f} %'.format(acc/10))