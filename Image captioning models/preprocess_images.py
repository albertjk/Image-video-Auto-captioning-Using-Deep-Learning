# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script is used for pre-processing images before training.
import tensorflow as tf
import os
from pickle import dump 

# Change this to get_flickr8k_data or get_mscoco_data or get_google_conceptual_data 
# for the Flickr8K, MSCOCO, and Google Conceptual Captions datasets, respectively.
from get_mscoco_data import read_set

def load_image(image_path):
	"""
	Loads an image, resizes it, and adapts it to the format
	which the CNN requires.
	Args:
		image_path: the path to the image
	Returns:
		img_array: the image in the preprocessed format
	"""

	# Load the image and resize it to what the model expects.
	# If using VGG19: target_size=(224, 224))
	# If using Xception: target_size=(299, 299))
	image = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))

	# Convert image pixels to a numpy array.
	img_array = tf.keras.preprocessing.image.img_to_array(image)

	# Reshape the image data for the model.
	# Reshape it to what the model expects as input.
	img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

	# Adapt the image format to what the model requires.
	img_array = tf.keras.applications.xception.preprocess_input(img_array)

	return img_array

def extract_features(set_name, set_images):
	"""
	Extracts the features of each image from the set_images list. 
	This list contains the images of the specified set.
	The features of each image are saved to a pickle file.
	Args:
		set_name: the name of the set - train or val (or test for Flickr8K)
		set_images: the list of images to extract features from
	"""
	
	# Load the CNN pre-trained on ImageNet images.
	model = tf.keras.applications.xception.Xception(weights='imagenet')

	# Remove the last layer (softmax output layer).
	# The last Dense layer will be the new output layer.
	model.layers.pop()
	model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)
	
	# If the directory for storing pre-processed images does not exist yet, create it.
	if not os.path.exists(extracted_features_dir + "/preprocessed_images/" + str(set_name) + "2014"):
		os.makedirs(extracted_features_dir + "/preprocessed_images/" + str(set_name) + "2014")

	# Extract features from each image.	
	image_count = 0
	for image_count, set_image in enumerate(set_images):		

		features = dict()
	
		# Load the image and adapt it to the format that the model expects.
		img = load_image(set_image)
		
		# Get the image features.
		feature = model.predict(img, verbose=0)
		
		# Keep only the actual image name.
		image_name = set_image.split("/")[-1] 
		
		# Remove the file extension.
		image_name = image_name.split('.')[0] 
		
		# Store the image features in a pickle file for each image.
		dump(feature, open(extracted_features_dir + "/preprocessed_images/" + str(set_name) + "2014/" + image_name + ".pkl", 'wb'))		

		print('Extracted features from image ' + str(image_count))
		print('> %s' % image_name)

	print("Image features are saved as pickle files.")

if __name__ == '__main__':   

	# If the directory for storing extracted image-caption data does not exist yet, create it.
	extracted_features_dir = "extracted_features_mscoco_Xception"
	if not os.path.exists(extracted_features_dir):
		os.makedirs(extracted_features_dir)

	# Set the size of the training set and the validation set to be used.
	train_set_size = 15000
	val_set_size = 2000

	# Get captions and images of the training and the validation sets.
	# If using a limited dataset, set the last parameter False. 
	# If using the full dataset, set it True.
	# For MSCOCO and Conceptual Captions, the second set is called 'val'.
	# For Flickr8K, this set is called 'test'.
	train_captions, train_images = read_set('train', train_set_size, False)
	val_captions, val_images = read_set('val', val_set_size, False)

	print("length of train_images:", len(train_images))
	print("length of train_captions:", len(train_captions))
	print("length of val_images:", len(val_images))
	print("length of val_captions:", len(val_captions))

	# Extract features from training images.
	print("Extracting features from training images...")
	extract_features("train", train_images)

	# Extract features from validation images.
	print("Extracting features from validation images...")
	extract_features("val", val_images)
