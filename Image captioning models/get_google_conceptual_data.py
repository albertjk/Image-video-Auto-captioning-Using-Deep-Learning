# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script gets captions and images of the Google Conceptual Captions dataset.
from sklearn.utils import shuffle

def read_set(set_name, limit, use_full_dataset):
	"""
	Reads the image ID and caption pairs of the specified set (train or val) of the Google Conceptual 
	Captions dataset. If limit is not zero, it limits the number of these pairs read. 
	The lists to be returned contain captions and image IDs, respectively, where 
	each caption and image ID pair is stored at the same index.
	Args:
		set_name: train or val set of the dataset
		limit: the number of elements to keep from the specified set
		use_full_dataset: boolean to decide if the entire set is to be returned. If True, the limit value is not taken into account.
	Returns:
		captions_of_set: a list storing limit amount or all captions of the specified set
		image_ids_of_set: a list storing limit amount or all image IDs of the specified set
	"""
	
	# These lists will store the captions and image IDs, respectively, each at the same index.
	captions_of_set = []
	image_ids_of_set = []	

	# Read the text file storing image IDs and their associated captions of the specified set.
	filename = "google_conceptual_data/" + str(set_name) + "_captions.txt"
	with open(filename) as f:
		lines = f.readlines()	
	
	for line in lines:
		
		# Each image ID to caption mapping is separated by a ':' symbol, so split each line.
		line = line.split(":")		

		# Get the image ID.
		image_id = "google_conceptual_data/" + str(set_name) + "_images/" + line[0] + ".jpg"
			
		# Get the caption. Remove trailing newline characters.
		caption = str(line[1:]).rstrip()		
			
		# Produce a single string of the caption tokens.
		caption = ' '.join(line[1:]).rstrip()
				
		image_ids_of_set.append(image_id)
		captions_of_set.append(caption)
			
	image_ids_of_set, captions_of_set = shuffle(image_ids_of_set, captions_of_set, random_state=1)
	
	# If not using the full dataset, use only the specified subset.
	if not use_full_dataset: 
		captions_of_set = captions_of_set[:limit]
		image_ids_of_set = image_ids_of_set[:limit]
	
	return captions_of_set, image_ids_of_set
