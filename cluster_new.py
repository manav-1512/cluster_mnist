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

# Applying Kmeans Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 10, init='k-means++', n_init=1, random_state=0)
y_kmeans = kmeans.fit_predict(data)

# Loading the pretrained CNN model used to provide with labels
model = keras.models.load_model('mnist.model')

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

acc = 100* np.sum(np.amax(y, axis = 0)) / y_kmeans.shape[0]


# Printing accuracy of KMenas Clustering
print('Accuracy = {:.2f} %'.format(acc))
