# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script is used for training the optimised model on the Google Conceptual Captions dataset.
import tensorflow as tf
import numpy as np
from pickle import load
import time
import os
import matplotlib.pyplot as plt

from preprocess_captions import get_captions_from_dict
from train_baseline import get_img_to_cap_dict, create_sequences

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

def build_model(vocab_size, max_len, optimizer):
	"""
	Creates the optimised image captioning model best suited for the Conceptual Captions dataset.
	The model consists of a pre-trained Xception CNN 
	as the encoder, and a decoder RNN.
	Args:
		vocab_size: the number of words in the vocabulary of the training captions
		max_len: the length of the longest caption in the training set
	Returns:
		model: the built model
	"""

	# Layer for getting the image features as input.
	cnn_fc_input = tf.keras.layers.Input(shape=(2048,))
	
	# Apply batch normalization on this Dense layer.
	fully_connected = tf.keras.layers.Dense(512, use_bias=False)(cnn_fc_input) 
	batch_norm = tf.keras.layers.BatchNormalization()(fully_connected)
	activation = tf.keras.layers.Activation('relu')(batch_norm)
    
	# This dropout layer is added to prevent overfitting.
	dropout = tf.keras.layers.Dropout(0.5)(activation) 
	
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
	
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print(model.summary())
    
	return model

if __name__ == '__main__':   

	# The path where the extracted features are saved.
	extracted_features_dir = "extracted_features_google_300K_full_Xception"
    
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

	epochs = 50
	
	# Experiments showed that the AdaMax optimizer with this learning rate works best. Decay rate also helps.
	# Batch normalization is part of the model as it enhanced performance during experiments on Conceptual Captions. 
	lr = 0.019
	decay_rate = lr / epochs
	optimizer = tf.keras.optimizers.Adamax(lr=lr, decay=decay_rate)	
        
	model = build_model(vocabulary_size, train_caps_max_length, optimizer)		
	
	# If the directory for saving trained models does not exist yet, create it.
	if not os.path.exists("trained_models"):
		os.makedirs("trained_models")

	file_path = 'trained_models/Optimised_Google.h5'

	# Name of the model to see on TensorBoard.
	model_name = "Optimised_Google"

	# Replace each semicolon with a dash so a new directory can be created storing the TensorBoard log.
	model_name = model_name.replace(":", "-")

	print("Model being trained:",model_name)

	# Callback for writing the TensorBoard log during training.
	callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='trained_models/tensorboard_logs/' + str(model_name), histogram_freq=0, write_graph=False)

	# Callback for writing checkpoints during training.
	callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	# Callback to apply early stopping if the model's validation loss does not decrease for 15 consecutive epochs.
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=15)

	callbacks = [callback_checkpoint, callback_tensorboard, early_stopping_callback]
    
    # Prints a photo of the architecture of the model.
	tf.keras.utils.plot_model(model, to_file="trained_models/optimised_google_model.png", show_shapes=True)

	model.fit([X1_train, X2_train], y_train, batch_size=512, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=([X1_val, X2_val], y_val))

    
