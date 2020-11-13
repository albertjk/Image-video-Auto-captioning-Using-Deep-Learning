# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script can generate a caption for test images using a trained image captioning model.
import tensorflow as tf
import numpy as np
from pickle import load
import os
import matplotlib
matplotlib.use("agg") # Backend to matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from preprocess_captions import get_captions_from_dict
from train_baseline import get_img_to_cap_dict

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

def extract_features(image):
	"""
	Extracts the features of the input image. 
	Returns the extracted features.
	Args:
		image: the image to extract features from
	Returns:
		features: the extracted image features
	"""
	
	# Load the pre-trained CNN.
	model = tf.keras.applications.xception.Xception(weights="imagenet")

	# Remove the last layer (softmax output layer).
	# The last Dense layer will be the new output layer.
	model.layers.pop() 
	model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-1].output)
	
	# Load the image and adapt it to the format that the model expects.
	img = load_image(image)
		
	# Get the image features.
	features = model.predict(img, verbose=0)

	return features
	
def index_to_word(searched_index, tokenizer):
	"""
	Takes an input integer index and returns the word it is mapped to.
	Args:
		searched_index: the integer index of the searched word
		tokenizer: the tokenizer which contains the word-index mappings
	Returns:
		word: the actual string word that the index is mapped to
	"""

	for word, integer in tokenizer.word_index.items():
		if integer == searched_index:
			return word 

	return None

def generate_caption(model, tokenizer, image, max_length):
	"""
	Generates a caption for the input image.
	Args:
		model: the trained image captioning model that generates the caption words
		tokenizer: the tokenizer trained on the captions of the training set
		image: the features of the input image 
		max_length: the maximum length of the caption to be generated
	Returns:
		the generated caption without the special start and end tokens
	"""

	# Begin with the start token, and append words to the input text.
	input_text = ["<start>"]

	# Repeatedly add words to the caption sentence.
	for i in range(max_length):

		# Encode the input text into integers.
		# Create a word to integer index mapping.
		encoded_text_sequence = tokenizer.texts_to_sequences([input_text])[0]

		# Pad each input text sequence to the same length.			
		encoded_text_sequence = tf.keras.preprocessing.sequence.pad_sequences([encoded_text_sequence], maxlen=max_length)
        
		# Predict the upcoming word in the caption sentence.
		# This returns an array of probabilities for each vocabulary word.
		predictions = model.predict([image, encoded_text_sequence], verbose=0)
        
		# Get the index of the largest probability - the index of the most likely word.
		index = np.argmax(predictions)

		# Get the word associated with the index.
		word = index_to_word(index, tokenizer)

		# If the index cannot be mapped to a word, stop.
		if word is None:
			break

		# Add the textual word as input for generating the next word.
		input_text.append(word)

		# If the end of the caption is predicted, stop.
		if word == '<end>':
			break
            
	# Exclude the start and end caption markers from the generated caption.
	final_caption = []
	for w in input_text:
		if w != '<start>' and w != '<end>':
			final_caption.append(w)
            
	# Create a string of the caption words.  
	caption_string = ' '.join(final_caption[:])

	return caption_string
        
if __name__ == '__main__':

	# The path where the extracted features are saved.
	extracted_features_dir = "extracted_features_mscoco_Xception"

	# Get the tokenizer.
	tokenizer = load(open(extracted_features_dir + '/train_tokenizer.pkl', 'rb'))
    
	# Print the vocabulary size to provide information to the developer.
	vocabulary_size = len(tokenizer.word_index) + 1
	print('Vocabulary size:',vocabulary_size)

	# Get the maximum caption length. This consists of the following steps:

	# Load the image to caption mappings of the training set.
	train_img_to_cap_dict = get_img_to_cap_dict(extracted_features_dir + '/train_captions.txt')

	# Extract all training captions from the train_img_to_cap_dict dictionary and store them in a list.
	train_captions = get_captions_from_dict(train_img_to_cap_dict)

	# Get the number of words in the longest caption of the training set.
	# The generated caption for the test images will be of this length maximum.
	train_caps_max_length = max(len(cap.split()) for cap in train_captions)
	print("Max caption length:", train_caps_max_length)

	# Load the trained image captioning model.
	# Change the path depending on which model to load.
	model = tf.keras.models.load_model('trained_models/Optimised_MSCOCO.h5')

	# Load the test images and generate a caption for each.
	directory = 'google_test_images'
	images = [os.path.join(directory, img) for img in os.listdir(directory)]
	images = sorted(images)

	for image in images:

		print("Image:", image)
            
		# Uncomment these lines to plot each image.
		#img = mpimg.imread(image)
		#imgplot = plt.imshow(img)
		#plt.show()

		image_features = extract_features(image)
		caption = generate_caption(model, tokenizer, image_features, train_caps_max_length)
		print(caption)                                                                                                                                                            