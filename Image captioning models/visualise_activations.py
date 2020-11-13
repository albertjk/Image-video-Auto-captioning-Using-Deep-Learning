# Author: Albert Jozsa-Kiraly
# Project: Image/video auto-captioning using Deep Learning

# This script can be used to visualize the activations of the VGG19 ConvNet
# using an example image.
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

def load_image(image_path):
	"""
	Loads an image, resizes it, and adapts it to the format
	which the CNN requires.
	Args:
		image_path: the path to the image
	Returns:
		img_array: the image in the preprocessed format
	"""

	# Load the image and resize it to what the VGG19 model expects.
	image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

	# Convert image pixels to a numpy array.
	img_array = tf.keras.preprocessing.image.img_to_array(image)

	# Reshape the image data for the model.
	# Reshape it to what the model expects as input.
	img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

	# Adapt the image format to what the model requires.
	img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

	return img_array

if __name__ == '__main__':

	# Load an example image that will be input to the model.
	image_path = 'mscoco_data/train2014/COCO_train2014_000000000081.jpg'
	image = load_image(image_path)

	# Load the pre-trained VGG19.
	model = tf.keras.applications.vgg19.VGG19()

	# Get the outputs of the layers.
	layer_outputs = [layer.output for layer in model.layers]

	# Create a model that, given the image input, returns the values of the layer activations.
	activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

	# Returns a list of numpy arrays: one array per layer.
	activations = activation_model.predict(image)

	# Plot the activations of the image across each layer.

	layer_names = []
	for layer in model.layers:

		# Store the names of the layers, so they can be added to the plot.
		layer_names.append(layer.name)
		
	images_per_row = 12

	# Display each feature map. The input layer is empty, so it is not displayed.
	for layer_name, layer_activation in zip(layer_names[1:], activations[1:]):

		# This variable stores the number of features in the feature map.
		num_of_features = layer_activation.shape[-1]
		
		# The feature map has a shape (1, size, size, num_of_features)
		size = layer_activation.shape[1]
		
		num_of_columns = num_of_features // images_per_row
		
		display_grid = np.zeros((size * num_of_columns, images_per_row * size))
		
		# Tile each filter into a big horizontal grid.
		for column in range(num_of_columns):
			for row in range(images_per_row):
				channel_image = layer_activation[0, :, :, column * images_per_row + row]
			
				# Make the feature visually satisfying.
				channel_image -= channel_image.mean()
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				display_grid[column * size : (column + 1) * size, row * size : (row + 1) * size] = channel_image
		
		scale = 1./size
		plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect='auto', cmap='viridis') # cmap='gray'
		plt.show()
		
	