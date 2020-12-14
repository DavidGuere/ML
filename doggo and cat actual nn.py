import tensorflow as tf
import pickle
import numpy as np
import time

############################################# importing Tensorboard ##########################################
# it is recommended to use a catchy name and specify the type of network and the number of neurons in each + the time to make it unique
name = f'Cat_vs_dog_cnn-70-70_nn-64-1_{int(time.time())}'
# this creates a directory named logs and inside a file named as we set above "name"
# we have to add this into the training and 
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{name}')
# To open the tensorboard we use the terminal set in the directory where the logs is created. We write:
# tensorboard --logdir=logs/
# We open the link with a browser

############################################# importing data ##########################################
with open('CatDogTraining_in.pkl', 'rb') as file:
    training_input = pickle.load(file)

with open('CatDogTraining_out.pkl', 'rb') as file2:
    training_output = pickle.load(file2)

############################################# normalizing data ##########################################
training_input = tf.keras.utils.normalize(training_input, axis=1)
# transforming to numpy array for validation
training_input = np.array(training_input)
training_output = np.array(training_output)
# shape [24944, 50, 50, 1]. We have 24944 samples, the actual size of the picture is 50x50x1.
# 1 is for the number of color channels. if RGB then it is 3, for monochrome is 1.

############################################# building the model ##########################################

model = tf.keras.models.Sequential()
# For convolutional networkk: call with .Conv2D(number_of_neutrons, kernel, activation_function)
# Kernel is the window (can be a matrix) that is gonna do the convolution in the data. In case of a matrix we have to enter (n,m)
# We can add padding to the image. Padding ads a layer of 0 around the image, this was when the kernel runs on the picture we
# dont lose on size and information. padding='same' adds a padding to the image and the image after the layer remains the same size
#                                    padding='valid' is the default setting. this DOES NOT ADD a padding
# If our input layer is a convolutional layer, we also have to specify the shape of the data:
# .Conv2D(number_of_neutrons, kernel, input_shape, avtivatoin_function, padding)
model.add(tf.keras.layers.Conv2D(70, (3, 3), input_shape=training_input.shape[1:], activation=tf.nn.relu,
                                 padding='same'))  # input layer
# We can also add a Max pooling layer. Max pooling runs a kernel through the picture and takes the maximum value of the
# values inside the kernel (pool size). it then slides to the next section of the picture. This will reduce the size of the image
# reducing computational power by allowing the network to see a simplified version of the picture. It also reduces the
# probability of overfiting.
# .MaxPooling2D(pool_size, shift)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(100, (3, 3), activation=tf.nn.relu, padding='same'))  # 1st layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(100, (3, 3), activation=tf.nn.relu, padding='same'))  # 2nd layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(100, (3, 3), activation=tf.nn.relu, padding='same'))  # 3rd layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(70, (3, 3), activation=tf.nn.relu, padding='same'))  # 4th layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(tf.keras.layers.Conv2D(70, (3, 3), activation=tf.nn.relu, padding='same'))  # 5th layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

# We can also add a normal neural network at the end for fun. We need to flatten the input first.
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))  # 2nd layer
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # output layer

############################################# setting up model for training ##########################################
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

############################################# training the model ##########################################
# Here we will add a validation_split. This is the amount of the data that it will take from the input to validate the
# model. In this case we will take 20% of the pictures to validate the model. For this the input data must be a numpy array!
# We add callbacks=[] to optimize the model using the tensorboard :)
model.fit(training_input, training_output, epochs=10, batch_size=30, verbose=2, validation_split=0.15,
          callbacks=[tensorboard])
print(model.summary())

############################################# Saving the model ##########################################
# name_of_model.save('name_we_want.model')
model.save('DoggoCat.model')
