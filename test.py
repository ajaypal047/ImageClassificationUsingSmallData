#numpy liberary for math and matrix operations
import numpy as np

# Keras image processing library for image manipulation and data reading 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.models import Sequential

from keras import backend as K

#importing Keras structures for standard layers
from keras.layers import Dropout, Flatten, Dense

#for using pretrained deep learning model i.e. VGG16
from keras import applications

#convert the labels 
from keras.utils.np_utils import to_categorical

#To plot lost function for observing error function convergence 
import matplotlib.pyplot as plt

import tensorflow as tf
#openCV for image reading and storing image
import cv2

#to parse command line argument for flags
import argparse

#to scan directory for getting the file list

from os import listdir, path
from os.path import isfile, join


# dimensions for rescaling the image to fit VGG
img_width, img_height = 224, 224


# pretrained model weights to use the transfer learning
top_model_weights_path = 'bottleneck_fc_model.h5'

# from tensorflow.python.client import device_lib

# print("using device       ",device_lib.list_local_devices())

#training data
train_data_dir = './train_61326'

#validation data
validation_data_dir = './val_61326'

# number of epochs 
epochs = 50

# batch size
batch_size = 32 


#standardization of bool inputs from command line Arguments
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#function to save output of VGG in .npy files for later use, storing data in .npy saves the calculation and provides freezing of VGG layers
def saveOutputs():
    # build the VGG16 network
	
    model = applications.VGG16(include_top=False, weights='imagenet')

    #using ImageDataGenerator to scale and manipulate images
    datagen = ImageDataGenerator(rescale=1. / 255)

    #reading and preparing the input from the train directory
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    #evaluating information recieved with help of the datagen object
    train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    #predicting the output from the model using the generator object as input
    trainOutput = model.predict_generator(
        generator, train_samples)

    #saving the output tensor
    np.save('trainOutput.npy', trainOutput)
		
    print(np.array(trainOutput).shape)
    print(np.array(train_samples).shape)

    ##### Validation output storing: similar to the train sample ##########
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    validationOutput = model.predict_generator(
        generator, nb_validation_samples)

    np.save('validationOutput.npy',
            validationOutput)


#training the model on top of VGG (i.e. adding the last fully connected layers)
def train_top_model():

	#defining a ImageDataGenerator from the train directory 
	datagen_top = ImageDataGenerator(rescale=1. / 255)

	#getting the images from the directory
	generator_top = datagen_top.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical',
		shuffle=False)

	#number of classes 
	num_classes = len(generator_top.class_indices)

	# save the class indices 
	np.save('class_indices.npy', generator_top.class_indices)

	# load the train output for VGG
	train_data = np.load('trainOutput.npy')
	
	train_data = train_data

	# get the class lebels for the training data, in the original order
	train_labels = generator_top.classes

	train_labels = to_categorical(train_labels, num_classes=num_classes)




	#do the similar steps for the validation set
	generator_top = datagen_top.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	nb_validation_samples = len(generator_top.filenames)

	validation_data = np.load('validationOutput.npy')

	validation_labels = generator_top.classes
	validation_labels = to_categorical(validation_labels, num_classes=num_classes)

	#building the last fully connected layers
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='sigmoid'))

	#Compiling the model
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	
	with tf.device('/gpu:0'):
		history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels))
		print("step Done")
		# saving the weights
		model.save_weights(top_model_weights_path)

		# evaluating on the evaluation set
		(eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

		plt.figure(1)
		# plot the accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()


def buildModel(image, class_dictionary):
         # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')
        num_classes = len(class_dictionary)

        # get the prediction from the pre-trained VGG16 model
        VGGprediction = model.predict(image)

        # build additionl fully connected layers
        model = Sequential()
        model.add(Flatten(input_shape=VGGprediction.shape[1:]))
        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.load_weights(top_model_weights_path)

        return model.predict_classes(VGGprediction)

def predict(predictDir):
    # load the class_indices saved in the earlier step

	imgFiles = [join(predictDir, f) for f in listdir(predictDir) if isfile(join(predictDir, f))]

	class_dictionary = np.load('class_indices.npy').item()

	for image_path in imgFiles:
		labels = []
		img = cv2.imread(image_path)
		img = cv2.resize(img,(3136,3136))
		for i in (2,3,4,5,6):
			for j in (0,1):
				image = img[i*448:(i+1)*448-1, j*448:(j+1)*448-1]

				#print("[INFO] loading and preprocessing image...")
				image = cv2.resize(image,(224, 224))
				image = img_to_array(image)

				# important! otherwise the predictions will be '0'
				image = image / 255

				image = np.expand_dims(image, axis=0)

				#final class
				class_predicted= buildModel(image, class_dictionary)

				inID = class_predicted[0]

				inv_map = {v: k for k, v in class_dictionary.items()}

				label = inv_map[inID]
				
				labels.append(label)
		
		print("Image ",image_path, "labels", labels)


def main():
    parser = argparse.ArgumentParser(description="Please Provide Input")
    parser.add_argument('--train', type=str2bool, default= False, required=False, help = 'set True if training')
    parser.add_argument('--saveVGG', type=str2bool, default= False, help='If to save the VGG data')
    parser.add_argument('--predictDir', type=str, default=None, help='Directory for the test Images')
    args = parser.parse_args()
    if(args.train):
        if(args.saveVGG):
            saveOutputs()
        train_top_model()
    elif(args.predictDir):
        if(path.isdir(args.predictDir)):
            print("Predicting the class:")
            predict(args.predictDir)
        else:
            print("ERROR: the dirctory provided for test images does not exist.")
    else:
        print("ERROR: please provide the testImage directory.")

    
if _name_ == '_main_':
    main()
