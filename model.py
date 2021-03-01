import os
import csv
import cv2
import numpy as np
import math
import matplotlib.image as mpimg
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

data_path = '../../data_collect/'

# PROCESS THE DATA IN BATCHES
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # data augmentation
            augmented_images = []
            augmented_angles = []
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle *-1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)



# READ IN DATA
lines = []

with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[3] == 'steering':
            continue
        lines.append(line)

images = []
measurements = []
correction = 0.2

for line in lines:
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    img_center = mpimg.imread(line[0])
    img_left = mpimg.imread(line[1])
    img_right = mpimg.imread(line[2])

    images.append(img_center)
    images.append(img_left)
    images.append(img_right)
    measurements.extend([steering_center])
    measurements.extend([steering_left])
    measurements.extend([steering_right])

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)


# AUGMENT THE DATA
augmented_images = []
augmented_measurements = []

for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)


# MODEL
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu', padding='valid'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu', padding='valid'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu', padding='valid'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# FIT THE MODEL
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Set our batch size
batch_size=32
epochs = 7

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')

# train model on all data
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=32, epochs=epochs, verbose=1)

# train and validate the model on only the center images
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=epochs, verbose=1)



# SAVE THE MODEL
print('Saving Model')
model.save('model.h5')