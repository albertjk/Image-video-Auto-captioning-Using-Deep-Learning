# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script downloads a specified number of validation images
# of the Google Conceptual Captions dataset, gets their captions, 
# and produces a text file of image ID to caption mappings.
# The 'Validation_GCC-1.1.0-Validation.tsv' file must be in the current directory.
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def download_images_and_get_captions(target):
	"""
	Downloads the images of the validation set using the links from the 
	validation TSV file. Images that are not downloaded successfully are removed.
	Writes the image ID and the caption of each downloaded image to a text file.
	Args:
		target: the number of images to download
	"""
	
	image_dir = 'val_images'
	
	# If the 'val_images' directory does not exist, create it.
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	captions = []

	# Store the image IDs and the associated captions in a new text file.
	captions_file = "val_captions.txt"
	g = open(captions_file, "w+", encoding='utf-8')

	# Open the data source file.
	validation_file = 'Validation_GCC-1.1.0-Validation.tsv'	
	with open(validation_file, encoding='utf-8') as f:
		lines = f.readlines()		

		# Continuously download images until there are as many 
		# in the 'val_images' folder as the target value.
		# Some images cannot be downloaded as the url is 
		# inaccessible, so remove these files. 
		link_line_count = 0
		downloaded_image_count = 0
		while len(os.listdir(image_dir)) != target:

			# Split the lines around tabs.		
			split_line = lines[link_line_count].split('\t')			

			# Get the link. Remove trailing characters including newline characters.
			url = split_line[1].rstrip()

			print('Donwloading image ' + str(downloaded_image_count) + '...')		

			try:			
				r = requests.get(url, allow_redirects=True)
				
			except Exception as e:
				print('The given URL could not be reached, so the image is skipped.')

			image_name = image_dir + '/' + str(downloaded_image_count) + '.jpg'

			open(image_name, 'wb').write(r.content)

			# Test if the image file was downloaded successfully. Try to open it. 
			# If there was an exception as it could not be opened, remove the faulty image, 
			# so it will not be part of the downloaded dataset. Skip its associated
			# caption too.
			downloaded_correctly = False
			try:
				img = mpimg.imread(image_name)
				imgplot = plt.imshow(img)
				downloaded_correctly = True
				print('Image ' + str(downloaded_image_count) + ' was downloaded successfully.')	

				print("Writing caption " + str(downloaded_image_count) + " to file...")

				# If the image was downloaded successfully, get the caption, 
				# and add it to the captions list.
				captions.append(split_line[0])

				# Only write the image ID and the corresponding caption to file if 
				# the image was downloaded successfully.
				# Each image ID will be mapped to its associated caption as follows:
				# Image index:caption
				g.write(str(downloaded_image_count) + ":" + str(captions[downloaded_image_count]) + "\n")
			except:
				print("Error. Image " + str(downloaded_image_count) + " cannot be opened as it was not downloaded correctly. Deleting image...")
				os.remove(image_name)

				# Keep the IDs (indexes) of downloaded images continuous.
				downloaded_image_count-=1

			downloaded_image_count+=1
			link_line_count+=1

	print('Images are downloaded.')
	print("Captions are written to file.")

if __name__ == '__main__':  

	# Only the first 10,000 images and captions are needed.
	limit = 10000

	download_images_and_get_captions(limit)