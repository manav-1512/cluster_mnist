# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Generating images in required format for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images in required format for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('trainingSet',
                                                 color_mode='grayscale',
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Creating the Test set
test_set = test_datagen.flow_from_directory('testSet',
                                            color_mode='grayscale',
                                            target_size = (28, 28),
                                            batch_size = 32)

# Importing libraries that are required for a CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# Initialising the CNN
cnn = Sequential()

# Convolution Layer 1
cnn.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 1]))

# Pooling Layer 1
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Convolution Layer 2
cnn.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Flattening of layers of CNN
cnn.add(Flatten())

# Full Connection stage of CNN
cnn.add(Dense(units=128, activation='relu'))

# Output Layer
cnn.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = 64,
                  epochs = 10,
                  validation_data = test_set,
                  validation_steps = 64)
             
# Saving the trained CNN for future use             
cnn.save('mnist.model')             