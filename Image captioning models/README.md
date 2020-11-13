# Image captioning models

This directory contains the code for pre-processing images and captions, training the baseline and optimised image captioning models, evaluating the models, and generating captions for new test images. TensorFlow and Keras are used to create the models. The model architecture consists of a CNN encoder and an RNN decoder. Image features are taken from the CNN via the last dense layer and are fed into the RNN. The RNN uses the image features, the current text sequence, and the upcoming word in the sequence to generate a caption. 

## Baseline model architecture

A VGG19 CNN is used which is pre-trained, and an LSTM is used as the RNN. The Adam optimiser is used with the default learning rate.

## Optimised model architecture

Three optimised versions of the model were created, one for each of the datasets used.

A pre-trained Xception CNN is used by all optimised models.

Hyperparameters of the model optimised for MSCOCO: AdaMax optimiser, 0.0145 learning rate with time-based decay, and batch normalisation applied.<br>
Hyperparameters of the model optimised for Flickr8K: AdaMax optimiser, 0.01604956256058524 learning rate with time-based decay.<br>
Hyperparameters of the model optimised for Conceptual Captions: AdaMax optimiser, 0.019 learning rate with time-based decay, and batch normalisation applied.

## Running the code

The code assumes that the relevant datasets are downloaded and placed in the following directories: `flickr8k_data`, `mscoco_data`, and `google_conceptual_data`. Before training a model, the `preprocess_captions.py` and the `preprocess_images.py` scripts must be run with the dataset specified in the code (either call `get_flickr8k_data.py` or `get_mscoco_data.py` or `get_google_conceptual data.py` in both `preprocess_images.py` and `preprocess_captions.py`). The number of images and captions used for training and validation must also be specified. For the MSCOCO and Conceptual Captions datasets, the two used sets are `train` and `val`. If using the Flickr8K data, the two sets are called `train` and `test`, which must be specified in the code. In addition, the output directory of the extracted features must be specified in the code. The `preprocess_images.py` script extracts image features from the training images and saves the features of each image to a pickle file. The `preprocess_captions.py` script cleans the captions, creates a tokenizer using the training captions and saves it to a pickle file, and writes the image to caption mappings to text files for both the training and validation sets.

After the data preparation, to train the baseline, the `train_baseline.py` script can be run with the directory of the pre-processed data and the output directory for saving models both specified in the code. To train the optimised model on the Flickr8K, MSCOCO, or Conceptual Captions dataset, the following scripts can be used: `train_optimised_flickr8k.py`, `train_optimised_mscoco.py`, or `train_optimised_google.py`, respectively. The directory of the pre-processed data and the output directory for saving models both need to be specified in the code. The models are defined in the training scripts. When training a model, the TensorBoard log is saved to the `tensorboard_logs` directory so that accuracy and validation scores can be viewed in TensorBoard after training. To evaluate a trained model, the `evaluate.py` script can be used with the path to the trained model and the extracted features both specified in the code. This evaluation script uses the BLEU metrics to evaluate model performance. To generate captions using a trained model, the `generate_caption.py` script can be used with the path to the trained model, the extracted features, and the test images all specified in the code. 

In addition, the visualisation script `visualise_activations.py` can be run to plot the ConvNet activations at different layers.

### Using CPU or GPU

This code is for training and evaluating models and generating image captions using a CPU since a regular LSTM layer is used as part of the models. To train and evaluate models and generate captions using a GPU, the `tf.keras.layers.LSTM` layer should be changed to `tf.keras.layers.CuDNNLSTM` which allows for faster training.
