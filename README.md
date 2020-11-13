# Undergraduate Dissertation Project - Image/video auto-captioning using Deep Learning

## Overview

This repository contains the code for my Computing Science undergraduate dissertation project I did in my final year at the University of Stirling. This project involved the creation of an image captioning model and a web interface. First a baseline model was created, then it was optimised to improve its captioning performance, and was trained on the Flickr8K, MSCOCO, and Google Conceptual Captions datasets. The trained model was integrated into a website which allows the user to browse a video to add it to a library and process it. This processing involves the extraction of video key frames, and the caption generation for each extracted frame using the trained model. The website has a search functionality which allows the user to find a specific search term in the captions of all library videos. Furthermore, the website allows the user to play a video whilst caption markers are displayed on the progress bar, and captions are shown in tooltips as well as text overlay. The complete system essentially allows for video storytelling. 

## Screenshots of the website's main functionalities

### Adding a video to the library and processing it

![screenshot1](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/blob/master/Images/screenshot1.png "Website Screenshot 1")

### Playing a library video

![screenshot1](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/blob/master/Images/screenshot2.png "Website Screenshot 2")

### Searching video captions

![screenshot1](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/blob/master/Images/screenshot3.png "Website Screenshot 3")


## Built With

* Python 3
* HTML 5
* CSS 
* JavaScript
* PHP 5
* FFmpeg

Python libraries used:

* TensorFlow
* Keras
* Scikit-learn
* NumPy
* NLTK
* Matplotlib
* Requests

## Getting Started

### Prerequisites

FFmpeg, as well as Python 3 and the above mentioned libraries must be installed on your machine. To visualise the convolutional neural network's activations, TensorFlow version 1.8 or later must be installed. Additionally, your machine should be able to run PHP code, for which [XAMPP](https://www.apachefriends.org/index.html) is required on Windows computers.

For model training, the Flickr8K, the MSCOCO, or the Google Conceptual Captions datasets must be downloaded and placed in the `flickr8k_data`, `mscoco_data`, and `google_conceptual_data` directories, respectively, in the same directory where the code for image captioning models is placed.

### Downloading the application

To get a local copy of the application up and running follow these simple steps.

First, clone the repository:

```
git clone https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning.git
```

### Downloading the Google Conceptual Captions dataset

Before training a model on Conceptual Captions, the dataset must be downloaded. To do this, follow the instructions of the [Readme file](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/tree/master/Download%20the%20Google%20Conceptual%20Captions%20dataset) in the `Download the Google Conceptual Captions dataset` folder within this repository.

### Training an image captioning model

To train the baseline or an optimised version of the image captioning model, follow the instructions of the [Readme file](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/tree/master/Image%20captioning%20models) in the `Image captioning models` folder within this repository. 

### Running the complete application

To run the complete application, follow the instructions of the [Readme file](https://github.com/albertjk/Image-video-Auto-captioning-Using-Deep-Learning/blob/master/Complete%20system) in the `Complete system` folder within this repository. 

