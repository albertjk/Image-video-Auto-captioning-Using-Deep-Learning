# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script gets captions and images of the Flickr8K dataset.
from sklearn.utils import shuffle

def read_set(set_name, limit, use_full_dataset):
	"""
	Reads the image ID and caption pairs of the specified set (train or test) of the Flickr8k dataset. 
	If limit is not zero, it limits the number of these pairs read. 
	The lists to be returned contain captions and image IDs, respectively, where 
	each caption and image ID pair is stored at the same index.
	Args:
		set_name: train or test set of the dataset
		limit: the number of elements to keep from the specified set
		use_full_dataset: boolean to decide if the entire set is to be returned. If True, the limit value is not taken into account.
	Returns:
		captions_of_set: a list storing limit amount or all captions of the specified set
		image_ids_of_set: a list storing limit amount or all image IDs of the specified set
	"""

	# These lists will store the captions and image IDs, respectively, each at the same index.
	captions_of_set = []
	image_ids_of_set = []	
	
	# Open the file storing the image IDs of the specified set.
	filename = "flickr8k_data/Flickr8k_text/Flickr_8k." + str(set_name) + "Images.txt"
	with open(filename) as f:
		set_image_list = f.readlines()
		
	# Remove the trailing newline character from each image ID.		
	for i in range(0, len(set_image_list)):		
		set_image_list[i] = set_image_list[i].rstrip() 		
		
	# Open the file storing the image ID to caption mappings of the whole dataset.
	captions_file = "flickr8k_data/Flickr8k_text/Flickr8k.token.txt"		
	with open(captions_file) as g:
		img_to_caption_list = g.readlines()
			
	# Each mapping is separated by a '#' symbol, so split each line.
	for img_to_caption in img_to_caption_list:
		img_to_caption = img_to_caption.split("#")
				
		# Get the image ID and the caption.
		image_id = img_to_caption[0]
		caption = img_to_caption[1].split("\t")[1]			
				
		# If the image is in the specified set, add it to the image_ids_of_set list.
		if image_id in set_image_list:

			image_id = "flickr8k_data/Flicker8k_Dataset/" + str(image_id)
				
			image_ids_of_set.append(image_id)
			captions_of_set.append(caption)

	image_ids_of_set, captions_of_set = shuffle(image_ids_of_set, captions_of_set, random_state=1)
					
	# If not using the full dataset, use only the specified subset.
	if not use_full_dataset: 
		captions_of_set = captions_of_set[:limit]
		image_ids_of_set = image_ids_of_set[:limit]
	
	return captions_of_set, image_ids_of_set
