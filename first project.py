import tensorflow as tf

mnist = tf.keras.datasets.mnist  # 28x28 resolution pictures of numbers from 0 to 9

######################################## in version 2 this might not be necessary ##################################
############################# this will be directly put in the model definition ####################################
# define hidden layers. It can be any random number
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# classes?
n_classes = 10
batch_size = 100
'''
####################################################################################################################


######################################## in version 2 this might not be necessary ##################################
# Instead of variables we use placeholders, which are variables with empty data that can be later filled with data
# tf.compat.v1.placeholder(data_type, shape, name)
# the shape of the matrix is not known sometime, this is why we define the first entry as "None". This allows us to have
# a free shape.
# the .compat allows us to get modules from the different versions (v1 and v2)
'''
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 784))
y = tf.compat.v1.placeholder(tf.float32)
'''

# Instead of calling an empty variable that we would fill later, we straight fill in the data to the variables
# input_train is the pictures of the numbers. It is stored as color arrays. output_train is the actual number
(input_train, output_train), (input_to_test, output_to_test) = mnist.load_data()

# we normalize the data, this makes it easier for the model to train.
input_train = tf.keras.utils.normalize(input_train,
                                       axis=1)  # This can also work: input_train.reshape(-1, 28*28).astype('float32') / 255
input_to_test = tf.keras.utils.normalize(input_to_test,
                                         axis=1)  # output_to_test.reshape(-1, 28*28).astype('float32') / 255
# the .reshape reshapes the matrix. In our case we cant a 1xN matrix. In our project we will do it later

####################################################################################################################


# model
######################################## in version 2 this might not be necessary ##################################
'''
def nn_model(data):
'''
# creating the weights and biases for the 3 layers and the output
# .random_normal_initializer creates random number from a normal distribution
'''
    hidden_layer_1 = {'weights': tf.Variable(tf.compat.v1.random_normal_initializer([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.compat.v1.random_normal_initializer([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.compat.v1.random_normal_initializer([n_classes]))}
'''
# performing the operation: sum[(input_data * weights) + biases]
# .add(a, b) sums up a + b. the .matmul() performs a multiplication of the arguments inside ().
# Our input data is the input_data that we will later define
'''
    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
'''
# we apply an activation function with the .nn module. The activation function is the .relu function and is it
# acted on the layer_1 tensor
'''
    layer_1 = tf.nn.relu(layer_1)  # this is the output of the first layer
'''
# For the second layer hte input_data is the output of the first layer (layer_1)
'''
    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)
'''
# In the output we dont sum up.
'''
    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']
'''
# we dont use an activation function for the output
'''
    return output
'''
# The Sequential([layer1, layer2, ... layer_n]) API maps 1 input to 1 output (convenient, but not flexible)
model = tf.keras.models.Sequential()

# To make the code clean, we dont add the layers in the (), but rather use the name_of_moder.add(layer) to add layers
# and avoid sending a list
model.add(tf.keras.layers.Flatten())  # flattens the input to a 1xN matrix

# .Dense(number_or_neurons_in_layer, activation_function, bias) defines a neural network. Performs the operation:
# sum[(input_data * weights) + biases]
model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))  # 1st layer
model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))  # 2nd layer
model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))  # 3rd layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output. Does not require activation function, but instead a prob. distrib. We can add the .softmax in the compilation of the model

####################################################################################################################


# training
######################################## in version 2 this might not be necessary ##################################

# Now we will tell tensorflow how to use the data in the previous model
'''
def train_neural_network(input_data):
'''
# We are gonna feed the input_data into our nn_network model
'''
    prediction = nn_model(input_data)
'''
# the cost function is a lost function for the entire batch. It also measures the deviance between the predicted
# and the actual value. The .reduce_mean computes the mean of elements across dimensions of a tensor. The
# softmax_cross_entropy_with_logits() is the cross function
'''
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
'''
# We want to cost function to be as small as possible, so we have to optimize it. We take the gradient and
# multiply it by a small number, the learning rate. We use the .AdamOptimizer(learning_rate).what_to_do(with_what)
# for optimizing. If the learning rate is empty, then it is set to 0.001. We set the optimizer to minimize
# with .minimize(what_we_want_to_minimize)
'''
    optimizer = tf.train.AdamOptimizer().minimize(cost_function)
'''
# We need to feed the data multiple times through the same neural network to optimize it. the number of times that
# we feed in the data is called epoch. Sometimes the data can be too big, so we divide that in batches. Each batch
# would be one iteration.
'''
    number_of_epochs = 10
'''
# we initialize the session
'''
    with tf.compat.v1.Session() as sess:
'''
# We initialize all variables with initialize_all_variables()
'''
        sess.run(tf.compat.v1.initialize_all_variables())
        
        for epoch in range(number_of_epochs):
            epoch_loss = 0
'''
# iterations. Divides the number of training examples in the library with the batch size
'''
            for could_be_the_number_of_iterations in range(int(mnist.train.num_examples / batch_size)):
'''
# separates the training examples into batches: .next_batch(the_batch_size)
'''
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
'''
# Tensoflow magically know that he has to optimize the cost function modifying the weights and biases
# in the hidden layer definition
'''
                could_be_the_number_of_iterations, cost = sess.run([optimizer, cost_function],
                                                                   feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += cost

            print('Epoch', epoch, 'completed out of the', number_of_epochs, 'loss', epoch_loss)
'''
# To check if the prediction is correct, we check if the prediction is equal to the input by looking for
# the maximum value in the tensor. If the position of the max value is the same, then both are equal.
# The .argmax(input, index_number) return the index of the max value of the input in the index_number axis.
# By default, the index_number is 0
# .equal(a, b) check is a and b are equal
'''
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
            print('Accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''
# training parameters
model.compile(optimizer=tf.keras.optimizers.Adam(),
              # here we use Adam(learning_rate) optimizer. We can specify the learning rate: lr=0.001, but default is good
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              # if we havent defined the .softmax() prob. func. before, we can add in the () from_logits=true
              metrics=['accuracy'])
# actual training
'''
train_neural_network(x)
'''
# to train, we use name_of_mode.fit(input_to_train, output_to_train, batch_size=number, epochs=number, verbose=number)
# verbose is to see the progress: 0 - nothing, 1 - progress bar, 2- on line per epoch
model.fit(input_train, output_train, batch_size=500, epochs=10, verbose=2)

# we have to be careful not to overfit. If we overfit, the model wont get a general ide, but memorize all the values.
# we want the model to learn patterns and attributes rather then memorize things (nem magolni)
# To check for overfitting we use
results = model.evaluate(input_to_test, output_to_test)
print("test loss, test acc:", results)
# the accuracy should be slightly lower than the one at the end of the training and the loss should be slightly higher
# we dont want to see too close of too big delta! this means overfiting
# 100 epochs is too much. 10 seems fine

# To save the model and use it later: model_name.save('the_name_that_we_want')
model.save('our_first_projetc.model')
'''
# To use it: variable = tf.keras.models.load_model('the_name_that_we_saved')
# This will give a bunch of "one hot" arrays in form of probability distribution
my_model = tf.keras.models.load_model('our_first_projetc')  # loads the model
first_prediction = my_model.predict([input_to_test])  # we have to give a list!

# To make it understandable, we have to take the index (the index maps to the predicted number) of the highest probability
# We du this with argmax
readable_solution = tf.argmax(first_prediction)
print(readable_solution[0])

# To check if it is the actual value
import matplotlib.pyplot as plt
plt.imshow(input_to_test[0])
'''