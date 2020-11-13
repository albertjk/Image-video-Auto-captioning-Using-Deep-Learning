# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script is used for training the baseline model.
import tensorflow as tf
import numpy as np
from pickle import load
import time
import os
import matplotlib.pyplot as plt

from preprocess_captions import get_captions_from_dict

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

		# Produce a string of the caption tokens.
		caption = ' '.join(caption)

		# If the image name is not in the dictionary yet, 
		# create a list to add captions of this image.
		if image_name not in img_to_cap_dict:
			img_to_cap_dict[image_name] = []

		img_to_cap_dict[image_name].append(caption)

	return img_to_cap_dict

def create_sequences(tokenizer, max_length, img_to_cap_dict, img_features, vocab_size):
	"""
	Creates input-output pairs for the model using the 
	extracted image features and captions of the specified set.
	Args:
		tokenizer: the tokenizer trained on captions of the training set
		max_length: the length of the longest caption of the training set
		img_to_cap_dict: the dictionary storing image to caption mappings
		img_features: the dictionary storing extracted image features
		vocab_size: the number of words in the vocabulary
	Returns:
		X1: image features which are input to the model
		X2: encoded current text sequence which is input to the model
		y: encoded next word in the text sequence which is the model output
	"""

	X1, X2, y = [], [], []

	# Loop over each dictionary entry.
	for image_name, captions_list in img_to_cap_dict.items():

		# Loop over each caption of the current image.
		for caption in captions_list:

			# Encode the caption text into integers.
			# Create a word to integer index mapping.
			encoded_caption = tokenizer.texts_to_sequences([caption])[0]

			# Split the encoded caption into multiple X, y pairs.
			for i in range(1, len(encoded_caption)):
				encoded_input_text_sequence, encoded_output_word = encoded_caption[:i], encoded_caption[i]

				# Pad each input text sequence to the same length.
				encoded_input_text_sequence = tf.keras.preprocessing.sequence.pad_sequences([encoded_input_text_sequence], maxlen=max_length)[0]
                
				# Get a binary matrix representation of the output word.
				encoded_output_word = tf.keras.utils.to_categorical([encoded_output_word], num_classes=vocab_size)[0]

				# Store the inputs and outputs in the corresponding lists.
				X1.append(img_features[image_name][0])
				X2.append(encoded_input_text_sequence)
				y.append(encoded_output_word)

	return np.array(X1), np.array(X2), np.array(y)

def build_model(vocab_size, max_len):
	"""
	Creates the baseline image captioning model.
	The model consists of a pre-trained VGG19 CNN 
	as the encoder, and a decoder RNN.
	Args:
		vocab_size: the number of words in the vocabulary of the training captions
		max_len: the length of the longest caption in the training set
	Returns:
		model: the built model
	"""

	# Layer for getting the image features as input.
	cnn_fc_input = tf.keras.layers.Input(shape=(4096,))
	
	fully_connected = tf.keras.layers.Dense(512, activation='relu')(cnn_fc_input) 
    
	# This dropout layer is added to prevent overfitting.
	dropout = tf.keras.layers.Dropout(0.5)(fully_connected) 
	
	# The RNN.
	rnn_input = tf.keras.layers.Input(shape=(max_len,))
	embedding = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=False)(rnn_input) 	
	lstm_layer = tf.keras.layers.LSTM(512)(embedding)	
	
    # Merge the two encoded inputs.
	dec = tf.keras.layers.add([dropout, lstm_layer])

	# This dropout layer is added to prevent overfitting.
	dropout2 = tf.keras.layers.Dropout(0.5)(dec) 

	outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(dropout2)   
	
	model = tf.keras.models.Model(inputs=[cnn_fc_input, rnn_input], outputs=outputs)
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	print(model.summary())
    
	return model

if __name__ == '__main__':   

	# The path where the extracted features are saved.
	extracted_features_dir = "extracted_features_flickr8k_vgg19"
    
	# Load the image features.
	train_features = get_all_features(extracted_features_dir + '/preprocessed_images/train2014')
	val_features = get_all_features(extracted_features_dir + '/preprocessed_images/val2014')

	# Load the image to caption mappings.
	train_img_to_cap_dict = get_img_to_cap_dict(extracted_features_dir + '/train_captions.txt')
	val_img_to_cap_dict = get_img_to_cap_dict(extracted_features_dir + '/val_captions.txt')

	# Get the tokenizer.
	train_tokenizer = load(open(extracted_features_dir + '/train_tokenizer.pkl', 'rb'))

	vocabulary_size = len(train_tokenizer.word_index) + 1
	print('Vocabulary size:',vocabulary_size)

	# Extract all training captions from the dictionary and store them in a list.
	train_captions = get_captions_from_dict(train_img_to_cap_dict)

	# Get the number of words in the longest caption of the training set.
	train_caps_max_length = max(len(cap.split()) for cap in train_captions)
	print("Max caption length:", train_caps_max_length)

	# Get input-output pairs for the model.
	X1_train, X2_train, y_train = create_sequences(train_tokenizer, train_caps_max_length, train_img_to_cap_dict, train_features, vocabulary_size)
	X1_val, X2_val, y_val = create_sequences(train_tokenizer, train_caps_max_length, val_img_to_cap_dict, val_features, vocabulary_size)

	# Create the baseline image captioning model.
	model = build_model(vocabulary_size, train_caps_max_length)
	
	# If the directory for saving trained models does not exist yet, create it.
	if not os.path.exists("trained_models"):
		os.makedirs("trained_models")

	file_path = 'trained_models/Baseline_Flickr8K.h5'

	# Name of the model to see on TensorBoard.
	model_name = "Baseline_Flickr8K"

	# Replace each semicolon with a dash so a new directory can be created storing the TensorBoard log.
	model_name = model_name.replace(":", "-")

	print("Model being trained:",model_name)

	# Callback for writing the TensorBoard log during training.
	callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='trained_models/tensorboard_logs/' + str(model_name), histogram_freq=0, write_graph=False)

	# Callback for writing checkpoints during training.
	callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	callbacks = [callback_checkpoint, callback_tensorboard]
    
	# Prints a photo of the architecture of the model.
	tf.keras.utils.plot_model(model, to_file="trained_models/model.png", show_shapes=True)

	model.fit([X1_train, X2_train], y_train, batch_size=1024, epochs=30, verbose=2, callbacks=callbacks, validation_data=([X1_val, X2_val], y_val))
    
