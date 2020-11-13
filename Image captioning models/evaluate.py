# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script evaluates a trained image captioning model using the BLEU scores.
import tensorflow as tf
import numpy as np
import os
from pickle import load
from nltk.translate.bleu_score import corpus_bleu

from preprocess_captions import get_captions_from_dict
from generate_caption import index_to_word

def get_img_to_cap_dict(filename):
	"""
	Opens the file storing image to caption mappings.
	Creates a dictionary and adds the mappings as 
	key-value pairs to it.
	Args:
		filename: the name of the file storing image to caption mappings
	Returns:
		img_to_cap_dict: the dictionary storing image to caption mappings
	"""

	file = open(filename, 'r')
	text = file.read()
	file.close()

	img_to_cap_dict = dict()

	for line in text.split('\n'):

		# Split each line by whitespace to get the image name and the caption.
		line_tokens = line.split()

		image_name, caption = line_tokens[0], line_tokens[1:]

		# If the image name is not in the dictionary yet, 
		# create a list to add captions of this image.
		if image_name not in img_to_cap_dict:
			img_to_cap_dict[image_name] = []

		img_to_cap_dict[image_name].append(caption)

	return img_to_cap_dict

def generate_caption(model, tokenizer, image, max_length):
	"""
	Generates a caption for the input image.
	Args:
		model: the trained image captioning model that generates the caption words
		tokenizer: the tokenizer trained on the captions of the training set
		image: the features of the input image
		max_length: the maximum length of the caption to be generated
	Returns:
		the generated caption which is wrapped around the special start and end tokens.
	"""

	# Begin with the start token, and append words to the input text.
	input_text = "<start>"

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
		input_text += " " + word

		# If the end of the caption is predicted, stop.
		if word == '<end>':
			break

	return input_text

def get_all_features(path_to_pkl_files):
	"""
	Collects all extracted features of images of the training or validation set
	stored in pickle files, and adds these features into one big dictionary.
	Args:
		path_to_pkl_files: the path where the pickle files are stored
	Returns:
		all_features: the dictionary storing the features of all images of the set
	"""

	all_features = dict()
	files = [os.path.join(path_to_pkl_files, f) for f in os.listdir(path_to_pkl_files)]
	for file in files:	       
        
		feature = load(open(file, 'rb'))

		# Remove the '.pkl' extension
		filename = file.split(".")[0]

		# Keep only the filename.
		filename = filename.split("\\")[-1]
	
		all_features[filename] = feature
	
	return all_features

def evaluate_model(model, tokenizer, max_length, img_to_cap_dict, img_features):
	"""
	Evaluates the captioning performance of a trained model against 
	a given set of ground truth captions. Calculates the BLEU scores.
	Args:
		model: the trained image captioning model 
		tokenizer: the tokenizer trained on the captions of the training set
		max_length: the maximum length of a caption
		img_to_cap_dict: the dictionary of image name to caption mappings
		img_features: the dictionary of image name to image features mappings
	"""

	ground_truth_captions, generated_captions = [], []

	# For each image name of the img_to_cap_dict dictionary,	
	# take the features of the current image and generate a caption. 
	# Store the generated caption and the ground truth captions for evaluation.
	for image, caption_list in img_to_cap_dict.items():

		generated_cap = generate_caption(model, tokenizer, img_features[image], max_length)

		actual_caps = []

		for cap in caption_list:
			actual_caps.append(cap)

		ground_truth_captions.append(actual_caps)
		generated_captions.append(generated_cap.split())

	# Calculate and print the BLEU scores for 1, 2, 3, and 4 cumulative n-grams using the appropriate weights.
	print('BLEU-1: %f' % corpus_bleu(ground_truth_captions, generated_captions, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(ground_truth_captions, generated_captions, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(ground_truth_captions, generated_captions, weights=(1.0/3, 1.0/3, 1.0/3, 0)))
	print('BLEU-4: %f' % corpus_bleu(ground_truth_captions, generated_captions, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == '__main__':  

	# The path where the extracted features are saved.
	extracted_features_dir = "extracted_features_flickr8k_vgg19"
    
	# Load the image features of the validation set.	
	val_features = get_all_features(extracted_features_dir + '/preprocessed_images/val2014')

	# Load the image to caption mappings from the text files.
	train_img_to_cap_dict = get_img_to_cap_dict(extracted_features_dir + '/train_captions.txt')
	val_img_to_cap_dict = get_img_to_cap_dict(extracted_features_dir + '/val_captions.txt')

	# Get the tokenizer.
	train_tokenizer = load(open(extracted_features_dir + '/train_tokenizer.pkl', 'rb'))

	# Extract all training captions from the train_img_to_cap_dict dictionary and store them in a list.
	train_captions = get_captions_from_dict(train_img_to_cap_dict)

	# Get the number of words in the longest caption of the training set.
	train_caps_max_length = max(len(cap) for cap in train_captions)
	print("Max caption length:", train_caps_max_length)

	# Load the trained image captioning model.
	# Change the path depending on which model to load.
	filename = 'trained_models/Baseline_Flickr8K.h5'
	model = tf.keras.models.load_model(filename)

	evaluate_model(model, train_tokenizer, train_caps_max_length, val_img_to_cap_dict, val_features)
