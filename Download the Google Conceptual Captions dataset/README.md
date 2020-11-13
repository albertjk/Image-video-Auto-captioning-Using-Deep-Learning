# Download the Google Conceptual Captions dataset

## Usage

The `get_train_captions_and_images.py` and `get_val_captions_and_images.py` scripts can be used for downloading the images of the training set and the validation set of the Google Conceptual Captions dataset, respectively. They require the `Train_GCC-training.tsv` and `Validation_GCC-1.1.0-Validation.tsv'` files to be in the current directory which can be downloaded from the [website of Google AI](https://ai.google.com/research/ConceptualCaptions/download). 

Before the download, the number of training and validation images to be downloaded must be specified in the scripts. While the scripts are downloading the images, captions for them are extracted from the two `tsv` files. The downloaded training images are placed into the `train_images` directory and the downloaded validation images are stored in the `val_images` directory. Two text files are created, `train_captions.txt` and `val_captions.txt` which store the image name to caption mappings line by line for the training set and the validation set, respectively. The downloaded dataset can be used for training image captioning models.
