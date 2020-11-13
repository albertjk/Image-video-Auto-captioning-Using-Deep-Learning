# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script is executed when the user has uploaded a video to the library to be
# processed. The processing involves the following steps: converting the video to mp4
# format if it is not in that format already, extracting key frames and getting their 
# times, generating a caption for each key frame, and saving each caption and its time 
# to a text file.
import glob
import subprocess
import os
import shutil
import sys
import numpy as np
from pickle import load

def convert_to_mp4(video_name):
	"""
	Uses an FFmpeg command to convert the input video to mp4 format
	which is then placed in the current working directory.
	Args:
		video_name: the name and format of the input video			
	"""
	
	new_video_name = str(video_name.split(".")[0])
	new_video_name = new_video_name + ".mp4"
	
	# Run the FFmpeg command to convert the video to mp4 format.
	command = ['ffmpeg', '-i', video_name, new_video_name, '-hide_banner']	
	subprocess.run(command)

def get_keyframes(video_path):
	"""
	Uses an FFmpeg command to extract key frames from the specified video file.
	Saves key frame times to a file.
	...
	Args:
		video_path: the path to the video file to extract key frames from
	Returns:
		key_frame_dir: the directory storing the extracted key frames
		key_frame_times: a list of the times associated with the key frames
	"""

	scene_change_threshold = '0.4'

	# The metadata of the key frames including their times in seconds will be stored in this text file.
	metadata_file = "key_frame_metadata.txt"

	print("Video path before splitting:",video_path)

	# Store the actual video name without the path and the file extension.
	video_name = video_path.split("\\")[-1]
	video_name = video_name.split(".")[0]

	print("Video name:",video_name)

	# If the directory for storing the key frames does not exist, create it outside the video_library directory.
	key_frame_dir = video_name + '_key_frames'
	if not os.path.exists(key_frame_dir):
		os.makedirs(key_frame_dir)

	output_file = key_frame_dir + '/img%03d.jpg'

	# Run the FFmpeg command to extract video key frames and store their metadata including their times in the text file.
	command = ['ffmpeg', '-i', video_path, '-filter:v', "select='gt(scene," + str(scene_change_threshold) + ")',metadata=print:file=" + metadata_file + "",'-vsync', '0', output_file]
	subprocess.run(command)

	# This array will store the time in seconds of each key frame.
	key_frame_times = []

	# Open the text file and read every line.
	with open(metadata_file, encoding='utf-8') as f:
		lines = f.readlines()
		
		# Get every second line as only those contain the times of key frames.
		for i in range(0, len(lines), 2):
		
			# Each line contains three elements: frame, pts, and pts_time.
			# Split around spaces and get pts_time which is the last element.
			split_line = lines[i].split(' ')			
			pts_time = split_line[len(split_line) - 1] 			
			
			# Remove the string "pts_time:" and strip new line characters from pts_time,
			# so only the actual time in seconds will remain.
			pts_time = pts_time[len('pts_time:') :].rstrip('\n')
			
			# Add 0.4s to pts_time so the markers are not placed on the progress bar early.
			pts_time = float(pts_time) + 0.4			
			pts_time = str(pts_time)

			print(pts_time)	
			
			# Make a copy of pts_time
			original_pts_time = pts_time	
			
			# Convert pts_time to this format: minutes : seconds. For example: 75 becomes 1:15 
			# The video player uses this time format.
			minutes = int(float(pts_time) / 60)
			seconds = float(original_pts_time) - (minutes * 60)
			
			converted_time = str(minutes) + ":" + str(seconds)			
			
			key_frame_times.append(converted_time)			
			
	print(key_frame_times)

	return key_frame_dir, key_frame_times

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
	
	# Load the CNN pre-trained on ImageNet images.
	model = tf.keras.applications.xception.Xception(weights='imagenet')

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

def get_generated_captions(key_frame_directory):
	"""
	Loads the specified trained image captioning model, 
	then loads each key frame and generates a caption for it.
	Args:
		key_frame_directory: the directory where all key frames are stored
	Returns:
		generated_captions: the list of generated captions for the key frames
	"""

	tokenizer = load(open('train_tokenizer.pkl', 'rb'))

	# Get the maximum caption length. This consists of the following steps:

	# Load the image to caption mappings of the training set.
	train_img_to_cap_dict = get_img_to_cap_dict('train_captions.txt')

	# Extract all training captions from the train_img_to_cap_dict dictionary and store them in a list.
	train_captions = get_captions_from_dict(train_img_to_cap_dict)

	# Get the number of words in the longest caption of the training set.
	# The generated caption for the key frames will be of this length maximum.
	train_caps_max_length = max(len(cap.split()) for cap in train_captions)
	print("Max caption length:", train_caps_max_length)

	# Load the trained image captioning model.
	model = tf.keras.models.load_model("Optimised_MSCOCO_FOR_DEMO.h5")
	print("Model loaded")

	generated_captions = []

	# Load the key frame images and generate a caption for each.
	directory = key_frame_directory
	images = [os.path.join(directory, img) for img in os.listdir(directory)]
	print("Generating captions for key frames...")
	for image in images:

		print("Image:", image)

		image_features = extract_features(image)
		caption = generate_caption(model, tokenizer, image_features, train_caps_max_length)

		print("Generated caption:")
		print(caption)
		
		generated_captions.append(caption)	
	
	return generated_captions
	
if __name__ == '__main__':

	# Find the most recently added file to the video library.
	list_of_videos = glob.glob("video_library/*")
	latest_video = str(max(list_of_videos, key=os.path.getctime))
	print("Latest video:",latest_video)

	path_to_latest_video = latest_video

	# Store the actual video name without the path.
	video_name = path_to_latest_video.split("\\")[-1]
	
	# If the video file is not in mp4 format, convert it to that format 
	# and store it in the video_library directory.
	file_extension = video_name.split(".")[1]
	if file_extension != 'mp4':
		convert_to_mp4(path_to_latest_video)		
		new_video_path = path_to_latest_video.split(".")[0] + ".mp4"
		
		# Continuously check if the converted video file exists. 
		# If so, the loop is finished.
		while not os.path.isfile(new_video_path):
			print("Waiting for",new_video_path)
			continue
			
		# Delete the video with the non-mp4 extension.
		os.remove(path_to_latest_video)
			
		# Update the path to the video with the mp4 extension.
		path_to_latest_video = new_video_path
		
		# Update the name of the video with the mp4 extension.
		video_name = path_to_latest_video.split("\\")[-1]
		
	# Store the actual video name without the file extension.
	video_name = video_name.split(".")[0]

	captions_file = "captions/" + video_name + ".txt"

	# If the caption file, associated with the video, does not exist, start processing the video.
	if(not(os.path.isfile(captions_file))):	
	
		print("Processing video...")		

		key_frame_dir, key_frame_times = get_keyframes(path_to_latest_video)

		# Check if there are extracted key frames.
		# The first key frame will be the thumbnail of the video.
		# It is copied to the "thumbnails" directory,
		# and is given the same name as the video.
		if len(os.listdir(key_frame_dir)) > 0:

			thumbnail_image = os.listdir(key_frame_dir)[0]
			source_location = key_frame_dir + "/" + thumbnail_image
			shutil.copy(source_location, "thumbnails")
			os.rename("thumbnails/" + thumbnail_image, "thumbnails/" + video_name + ".jpg")

			print("Generating captions...")					
			
			# Make sure the Python script can access all installed packages 
			# from the correct location. This sys.path.insert must be here 
			# to ensure that the script works.
			sys.path.insert(0, 'c:/users/albert jk/desktop/newenvs/newenvs/lib/site-packages')
			print(sys.path)

			# Due to running this Python script in a PHP script, this import is here 
			# just before tf functions are used. 
			import tensorflow as tf

			# Get the generated caption for each key frame.
			generated_captions = get_generated_captions(key_frame_dir)

			# After the captions are generated, the directory storing the key frames 
			# and the metadata file is deleted.
			shutil.rmtree(key_frame_dir)
			os.remove("key_frame_metadata.txt")

			# Write the time of each key frame to a text file.
			# There are as many key frames as generated captions.
			f = open(captions_file, "w+")
			print("Writing captions and start times to file...")			
			for i in range (len(key_frame_times)):
				print(generated_captions[i] + "#" + key_frame_times[i] + "\n")
				f.write(generated_captions[i] + "#" + key_frame_times[i] + "\n")

			f.close()

		print("Finished")