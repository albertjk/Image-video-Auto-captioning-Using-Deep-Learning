# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script gets captions and images of the MSCOCO dataset.
import os
import json
from sklearn.utils import shuffle

def read_set(set_name, limit, use_full_dataset):
	"""
	Reads the image name and caption pairs of the specified set (train or val) of the MSCOCO dataset. 
	If limit is not zero, it limits	the number of these pairs read. 
	The lists to be returned contain captions and image names, respectively, where 
	each caption and image name pair is stored at the same index.
	Args:
		set_name: train or val set of the dataset
		limit: the number of elements to keep from the specified set
		use_full_dataset: boolean to decide if the entire set is to be returned. If True, the limit value is not taken into account.
	Returns:
		captions_of_set: a list storing limit amount or all captions of the specified set
		image_names_of_set: a list storing limit amount or all image names of the specified set
	"""

	# These lists will store the captions and image names, respectively, each at the same index.
	captions_of_set = []
	image_names_of_set = []
    
	# Read the JSON file, which stores image name to caption mappings of the specified set.	
	annotation_file = "mscoco_data/annotations/captions_" + str(set_name) + "2014.json"
	with open(annotation_file, 'r') as f:
		annotations = json.load(f)
        
	# Get each caption and the corresponding image name, and add them to the appropriate lists.
	for annotation in annotations['annotations']:

		caption = annotation['caption']
		image_id = annotation['image_id']
		image_path = "mscoco_data/" + str(set_name) + "2014/COCO_" + str(set_name) + "2014_" + "%012d.jpg" % (image_id)
    
		captions_of_set.append(caption)
		image_names_of_set.append(image_path)
	print(len(image_names_of_set))

	captions_of_set, image_names_of_set = shuffle(captions_of_set, image_names_of_set, random_state=5)

	# If not using the full dataset, use only the specified subset.
	if not use_full_dataset: 
		captions_of_set = captions_of_set[:limit]
		image_names_of_set = image_names_of_set[:limit]

	return captions_of_set, image_names_of_set
