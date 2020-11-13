# Complete system

The complete system includes the trained image captioning model and the website for playing and searching captioned videos.

On the Home page of the website, the user can browse a video, which can be processed and added to their library of videos. On the Library page, the user can search the captions of all videos in the library. The user can enter a single word or a sentence consisting of multiple words as the search term. The results of the search are those videos that have associated captions that contain the search term. 

On click to a video thumbnail, that video is played, and all the search results are displayed in a playlist on the right side of the screen.  

Alternatively, when the user is on the Library page, they can click on a video to play it even without searching captions. The playlist shows the thumbnails of all videos in the library.  

On the Player page, the clicked video is automatically played. Red markers are displayed on the progress bar and each marker has an associated caption which describes the items that appear in the video at that scene (key frame). If the user hovers their mouse over a marker, the caption is displayed in a tooltip. When a captioned part of the video is reached, the caption is also displayed as a text overlay on the top of the video player.  

A button allows for skipping to the next marker and another button makes it possible to go back to the previous marker on the progress bar. Additionally, a search box allows searching the captions of the video being played by entering a single word or a sentence consisting of multiple words. If a caption contains the entered search term, the video player jumps to that marker. Underneath the player, there are caption buttons which represent the captions of the current video. On click to a caption button, the player jumps to the key frame described by that specific caption.

On the Home page, if the user clicks on the 'Process video and add to library' button, the browsed video is uploaded to the video library, then a Python script is launched which extracts key frames from the video using FFmpeg, then the trained image captioning model captions each key frame, and finally a text file is created which contains each generated caption and its associated time in the video. Additionally, the first key frame is used as the thumbnail of the video.

The image captioning model part of the system is the model optimised for the MSCOCO dataset. It uses a pre-trained Xception CNN as the encoder and an LSTM layer RNN as the decoder. It uses the AdaMax optimizer, 0.0145 learning rate with time-based decay, and batch normalisation. The model was trained on the full MSCOCO dataset and can be run on a CPU as it uses a regular LSTM layer and not CuDNNLSTM. The trained model file 'Optimised_MSCOCO_FOR_DEMO.h5' must be in the current directory, in order for the system to correctly operate. This file can be downloaded from https://drive.google.com/file/d/1jSj9SY4Uj8Sd_CBbqSGY3TlPC7UpjZVA/view. Alternatively, the file can be obtained by training the optimised model from scratch using the MSCOCO dataset.

The video player and the markers, the caption text overlay, and the two buttons 'Next marker' and 'Previous marker' are based on the VideoJS Marker plugin by [Samping Chuang](http://sampingchuang.com/videojs-markers).  

Used technologies:<br>
Image captioning model: Python, TensorFlow, Keras<br>
Video processing: FFmpeg
Website: HTML, CSS, JavaScript, PHP

## Usage

### Prerequisites

The XAMPP Control Panel (for PHP version 5) needs to be installed on your machine. 

### Running the code

Navigate to the directory where XAMPP is installed on your machine, and inside the `htdocs` folder create a new directory called `videoplayer`. Then, copy all files from the `Complete system` directory within this Github repository to the `videoplayer` folder on your machine. Run the XAMPP Control Panel, and start the Apache module (top module). To access the Home page, type the following in your browser's address bar: `http://localhost/videoplayer/home.html`