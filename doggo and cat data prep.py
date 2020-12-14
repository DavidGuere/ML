import numpy as np
import matplotlib.pyplot as plt
import os  # navigate, imoport, create files and stuff
import cv2  # image processing
import random  # to shuffle the data
import pickle  # to save the list

data_directory = 'C:\Anaconda3\Deep_learning\PetImages'
categories = ['Dog', 'Cat']

for animal in categories:
    path = os.path.join(data_directory, animal)  # creates the path for the Dog and Cat folder

    # now we will read the images and put it in an array
    for picture in os.listdir(path):  # os.listdir(path) lists all the files in 'path', that is, all the pictures
        # cv2.imread('path', option)  this reads the images from 'path' with the 'option'
        # the path will be the previously define "path", which is the path to the folders of the cat and dogs, and
        # the name of the image
        image_array = cv2.imread(os.path.join(path, picture), cv2.IMREAD_GRAYSCALE)

# The images are of different sizes. We have to normalize it. We use
size = 80
new_image_array = cv2.resize(image_array, (size, size))

# creating training data
training_data = []


def create_training_data():
    for animal_train in categories:
        path_train = os.path.join(data_directory, animal_train)
        class_num_of_cat_and_dog = categories.index(animal_train)  # classifying dog and cats as numbers

        for picture_train in os.listdir(path_train):
            try:
                image_array_train = cv2.imread(os.path.join(path_train, picture_train), cv2.IMREAD_GRAYSCALE)
                new_image_array = cv2.resize(image_array_train, (size, size))
                training_data.append([new_image_array, class_num_of_cat_and_dog])  # we assign the number that represents dog or cat to the array
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

# Now we are gonna separate the animals and the type of the animal into two arrays
training_input = []
training_output = []

for animal, fajta in training_data:
    training_input.append(animal)
    training_output.append(fajta)

# both are list. To feed it into a network is has to be numpy array
# print(type(training_input))
# print(type(training_output))

training_input = np.array(training_input).reshape(-1, size, size, 1)

# to save a file we use pickle. We have to open and close pickle, so we use "with".
# with open('name_of_file_to_create.pkl', 'wb') as random_name_to_open_the_file_in_numpy
#       pickle.dump(the_thing_we_want_to_save, random_name_to_open_the_file_in_numpy)

with open('CatDogTraining_in.pkl', 'wb') as random_file:  # we create an empty file named CatDogTraining_in and open it in numpy as "random_file"
    pickle.dump(training_input, random_file)  # we dump the data of training_input into random_file

with open('CatDogTraining_out.pkl', 'wb') as another_file:
    pickle.dump(training_output, another_file)

# to open a pickle file:
# with open('name_of_file_to_open.pkl', 'rb') as random_name_to_open_the_file_in_numpy:
#        variable_to_open = pickle.load(random_name_to_open_the_file_in_numpy)
with open('CatDogTraining_in.pkl', 'rb') as random_name_to_open_the_file_in_numpy:
    variable_to_open = pickle.load(random_name_to_open_the_file_in_numpy)

print(variable_to_open[:2])

