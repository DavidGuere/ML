# when the validation loss is going up, the validation accuracy will eventually fall. This means that the model is overfiting
# and it is starting to memorize samples rather then learning by generalizing (magol).
# The better the in-sample (training data) results are compared to the out-samples (validation data) the worse the model is.
# This means that the model is getting good only with the training samples rather than generalizing (magolás)
# There are many things that we can change in a model, but to start optimizing we start by changing the type of network
# (nn, cnn), and the number of neurons in each:

import tensorflow as tf
import pickle
import numpy as np
import time

############################################# importing data ##########################################
with open('CatDogTraining_in.pkl', 'rb') as file:
    training_input = pickle.load(file)

with open('CatDogTraining_out.pkl', 'rb') as file2:
    training_output = pickle.load(file2)

############################################# normalizing data ##########################################
training_input = tf.keras.utils.normalize(training_input, axis=1)
training_input = np.array(training_input)
training_output = np.array(training_output)

############################################# building the model ##########################################
############################################# Setting parameters for optimizing ##########################################
# it is recommended to chose numbers that are less and more than our first try
no_of_neurons = [40, 70, 100]
no_of_dense_layers = [0, 1, 2]
no_of_conv_layers = [1, 2, 3]

for dens_layer in no_of_dense_layers:
    for neurons in no_of_neurons:
        for conv_layer in no_of_conv_layers:
            ############################################# importing Tensorboard ##########################################

            name = f'Cat_vs_dog_NoOfCNN_{conv_layer}_NoOfNN_{dens_layer}_NoOfNeurons_{neurons}_Time_{int(time.time())}'
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'DogCatTrain/{name}')
            # to open in Centreboard: tensorboard --logdir=name/

            model = tf.keras.models.Sequential()

            # The inpute layer mus always be, so this will not change
            model.add(tf.keras.layers.Conv2D(neurons, (3, 3), input_shape=training_input.shape[1:], activation=tf.nn.relu, padding='same'))  # input layer
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

            # to create the other conv layer we need a for cycle
            for cl in range(conv_layer-1):
                model.add(tf.keras.layers.Conv2D(neurons, (3, 3), activation=tf.nn.relu, padding='same'))  # 1st layer
                model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

            # no matter how many dens layer we put, the output will always be a dense layer, so we need to flatten the output
            model.add(tf.keras.layers.Flatten())

            for dl in range(dens_layer):
                model.add(tf.keras.layers.Dense(neurons, activation=tf.nn.relu))  # 2nd layer

            model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))  # output layer

            ############################################# setting up model for training ##########################################
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

            ############################################# training the model ##########################################
            model.fit(training_input, training_output, epochs=10, batch_size=30, verbose=2, validation_split=0.1, callbacks=[tensorboard])
            print(model.summary())
