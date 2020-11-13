# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script is used for pre-processing captions before training.
import tensorflow as tf
import string
from nltk.tokenize import word_tokenize
from pickle import dump
import os
import time

# Change this to get_flickr8k_data or get_mscoco_data or get_google_conceptual_data 
# for the Flickr8K, MSCOCO, and Google Conceptual Captions datasets, respectively.
from get_mscoco_data import read_set
	
def get_image_to_caption_mappings(set_image_names, set_captions):
	"""
	Takes the image names and captions of the specified set,
	and creates a dictionary where each image name is associated
	with its list of captions. Returns this dictionary.
	Args:
		set_image_names: the list of image names of the specified set
		set_captions: the list of captions of the specified set
	Returns:
		set_image_to_caption_dict: the dictionary storing image name to caption mappings
	"""

	# This dictionary will store the image names mapped to captions.
	set_image_to_caption_dict = dict()

	# Loop over all image and caption pairs.
	for i in range(len(set_image_names)):
	
		image_name = set_image_names[i]		
		caption = set_captions[i]

		# If the current image is not in the dictionary yet,
		# create a list to store the captions of this image in the dictionary.
		if image_name not in set_image_to_caption_dict:
			set_image_to_caption_dict[image_name] = []

		# Add the caption to the dictionary.
		set_image_to_caption_dict[image_name].append(caption)

	return set_image_to_caption_dict
	
def clean_captions(img_to_cap_dict) :
	"""
	Takes as input the dictionary that stores image name to captions mappings.
	Cleans each caption and puts the cleaned caption back to the dictionary.
	Args:
		img_to_cap_dict: the dictionary which stores image name to captions mappings.
	"""
	
	# Loop over the dictionary items.
	for image, caption_list in img_to_cap_dict.items():
	
		# Convert each caption sentence to lowercase, tokenize it,
		# and remove punctuations and non-alphabetic characters.
		for i in range(len(caption_list)):

			caption = caption_list[i]
			
			caption = caption.lower()
			caption_tokens = word_tokenize(caption)	
			
			# This list will store the tokens of the cleaned caption.
			cleaned_caption = []			
			
			for token in caption_tokens:				
				if token not in string.punctuation and token.isalpha():		
					cleaned_caption.append(token)
					
			# Produce a string of the caption tokens.
			clean_caption = ' '.join(cleaned_caption)			
			
			# Wrap the caption between the special start and end tokens.
			clean_caption = "<start> " + clean_caption + " <end>"
			
			# Put the cleaned caption back to the dictionary.
			caption_list[i] = clean_caption	
			
def get_captions_from_dict(img_to_cap_dict):
	"""
	Extracts captions from the img_to_cap_dict dictionary
	and adds them to a list.
	Args:
		img_to_cap_dict: the dictionary storing image to caption mappings
	Returns:
		captions: a list storing all captions extracted from the dictionary
	"""

	captions = []

	for image_name, caption_list in img_to_cap_dict.items():
		for cap in caption_list:
			captions.append(cap)

	return captions

def save_img_to_caption_dict(img_to_cap_dict, filename):
	"""
	Saves each image name to caption mapping, one mapping per line, to a text file.
	Args:
		img_to_cap_dict: the dictionary storing image to captions mappings
		filename: the name of the output text file
	"""

	mappings = []	
	
	for image, caption_list in img_to_cap_dict.items():
		for cap in caption_list:
			mappings.append(image + ' ' + cap)
			
	file_content = '\n'.join(mappings)
	
	file = open(filename, 'w')
	file.write(file_content)
	file.close()
	
	print("'" + filename + "' saved")

def create_tokenizer(captions):
	"""
	Takes a list of captions and uses it to create a tokenizer,
	which is saved to disk.
	Args:
		captions: a list of captions to train the tokenizer on
	"""

	print("Creating Tokenizer...")

	# Keep the 10,000 most frequently used words.
	# Special characters are filtered except <> 
	# as the start and end caption markers are wrapped between these symbols.
	kept_words = 10000	

	# Use the words of training captions to create the tokenizer.
	train_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=kept_words, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
	train_tokenizer.fit_on_texts(captions)
	
	# Save the Tokenizer.
	dump(train_tokenizer, open(extracted_features_dir + '/train_tokenizer.pkl', 'wb'))
	
	print("'train_tokenizer.pkl' saved.")
	
	vocab_size = len(train_tokenizer.word_index) + 1
	print('Vocabulary size:',vocab_size)
    
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

	# Keep only the actual image names without the path.
	# Also remove the file extension.
	for i in range(len(train_images)):
		train_images[i] = train_images[i].split("/")[-1]
		train_images[i] = train_images[i].split('.')[0]
		
	for i in range(len(val_images)):
		val_images[i] = val_images[i].split("/")[-1]
		val_images[i] = val_images[i].split('.')[0]

	# Map each image of the training set to its associated captions. 
	img_to_caption_dict = get_image_to_caption_mappings(train_images, train_captions)
	print('Loaded the captions of', len(img_to_caption_dict), 'unique training images.')

	clean_captions(img_to_caption_dict)

	# Get all captions from the img_to_caption_dict dictionary and store them in a list.
	train_captions = get_captions_from_dict(img_to_caption_dict)

	create_tokenizer(train_captions)

	# Save the dictionary of image name to caption mappings to a file.
	# These mappings are needed when creating input-output sequences of training data to train the model.
	save_img_to_caption_dict(img_to_caption_dict, extracted_features_dir + '/train_captions.txt')

	# Map each image of the validation set to its associated captions. 
	img_to_caption_dict = get_image_to_caption_mappings(val_images, val_captions)
	print('Loaded the captions of', len(img_to_caption_dict), 'unique validation images.')

	clean_captions(img_to_caption_dict)

	# Save the dictionary of image name to caption mappings to a file.
	# These mappings are needed when creating input-output sequences of validation data.
	save_img_to_caption_dict(img_to_caption_dict, extracted_features_dir + '/val_captions.txt')
