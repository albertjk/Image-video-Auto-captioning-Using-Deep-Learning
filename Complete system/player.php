<?php
/**
* Author: Albert Jozsa-Kiraly
* Project: Image/video auto-captioning using Deep Learning
* 
* This script dynamically generates the content of the Video Player page 
* depending on the clicked video to play, items in the playlist, and whether 
* a search term was entered on the Library page.
*/

// $_POST['playlistButton'] stores the name of the video the user clicked on. This video will be played automatically.
// $_POST['playlistItems'] stores in an array the names of all videos that will be displayed as thumbnails in the playlist. 
// $_POST['librarySearchTerm'] stores the entered search term on the library page.   

// Check if the $_POST['playlistButton'] and $_POST['playlistItems'] are set and not null.	
if(isset($_POST['playlistButton']) && isset($_POST['playlistItems'])) {

	// Get the video name, and append the necessary characters to create the path to the video.
	$videoPath = 'video_library/' . $_POST['playlistButton'] . '.mp4';

	// Get the array storing the names of videos of the playlist.
	$playlistItems = $_POST['playlistItems'];
}

/* The user may not enter a search term on the Library page, so this check is separate from the previous check.
If a search term was entered on the Library page, get it. */
if(isset($_POST['librarySearchTerm'])) {		
	$librarySearchTerm = $_POST['librarySearchTerm'];
} else {
	$librarySearchTerm = null;
}
?>
<!doctype html>
<html>

<head>
	<title>Video Player page - Video captioning using Deep Learning</title>
	<meta charset="utf-8">
	<link rel="stylesheet" href="http://vjs.zencdn.net/4.3/video-js.css">
	<link rel="stylesheet" href="css/videojs.markers.css">
	<link rel="stylesheet" href="css/player.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
</head>
<body>

	<nav>
		<a href="home.html">Back to Home page</a>
		<a href="library.html">Back to video library</a>
	</nav>	

	<h1>Video captioning using Deep Learning</h1>
	
	<video id="test_video" controls preload="none" class="video-js vjs-default-skin" width="900" height="520">
		<source src="<?php echo $videoPath; ?>" type="video/mp4">
	</video>	

	<div class="control">
		<button type="button" id="videoNavButton" onclick="player.markers.next()">Next marker</button>
		<button type="button" id="videoNavButton" onclick="player.markers.prev()">Previous marker</button>
	</div>	

	<div class="search-box">
		<input id="search-txt" type="text" name="" placeholder="Type to search caption in video">
		<button type="submit" class="search-btn" onclick="searchCaption(document.getElementById('search-txt').value)">
				Search
		</button>
	</div>	

	<h4>Jump to specific captions of the video:</h4>
		
	<!-- Onload the web page, dynamically add videos to the playlist. -->
	<ul id="playlist">
	</ul>

	<!-- The captions of the video in the player (either all captions or just the results of a search) 
		will be dynamically added next to each other as buttons. If there are many buttons, the user can scroll horizontally. -->
	<div id="captionButtonsScroll">
	</div>
			
	<script src="js/video.js"></script>
	<script src="js/videojs-markers.js"></script>
	<script>
	var vid = document.getElementById("test_video");

		// Initialise video.js.
		var player = videojs('test_video');
		player.autoplay(true);

		// Load the marker plugin.
		player.markers({

			breakOverlay: {
				display: true,
				displayTime: 2,
				text: function(marker) {
					return marker.text;
				}
			},
			markers: []
		});

		// Wait for the metadata to load, otherwise the video duration cannot be received.
		vid.onloadedmetadata = function() {

			// Remove the captions of the previously played video.
			player.markers.removeAll();

			// Remove the caption buttons of the previously played video.
			var captionButtonsScroll = document.getElementById("captionButtonsScroll");
			while(captionButtonsScroll.firstChild) {
				captionButtonsScroll.removeChild(captionButtonsScroll.firstChild);
			}	

			/* If the user entered a search term on the Library page, only add the markers of those
			captions of the currently played video that contain this term. */
			var libSearchTerm = "<?php echo $librarySearchTerm; ?>";

			// Remove stop words from the search term.				
			var cleanLibSearchTerm = removeStopWords(libSearchTerm.toLowerCase());

			if(cleanLibSearchTerm != null && cleanLibSearchTerm != "") {
				addResultMarkers(getCaptionFileOfCurrentVideo(), cleanLibSearchTerm);
			} 
			/* Otherwise, there was no search on the Library page, 
			so add all markers of the captions of the currently played video. */
			else {
				addAllMarkers(getCaptionFileOfCurrentVideo());
			}				
		};

		/**
		 * Removes stop words from the input string,
		 * and returns the result string, which contains
		 * no stop words.
		 * @param {string} originalString: the original string which includes stop words
		 * @return {string} the string without stop words
		 */
		function removeStopWords(originalString) {

			// The list of stop words is taken from the NLTK Python module.
			var stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"];

			var originalTokens = originalString.toLowerCase().split(" ");
			var cleanTokens = [];
			for(var i = 0; i < originalTokens.length; i++) {
				stopWord = false;

				for(var j = 0; j < stopWords.length; j++) {
					if(originalTokens[i] == stopWords[j]) {
						stopWord = true;
						break;
					}	
				}

				if(!stopWord) {
					cleanTokens.push(originalTokens[i]);
				}
			}

			// Create a string of the clean tokens.
			var cleanString = cleanTokens.join(" ");

			return cleanString;
		}

		/**
		 * Reads a text file which contains captions and their start times.
		 * Puts the markers of those captions on the progress bar, at their specified start time, which contain the search term.
		 * @param {string} file: The text file to read
		 * @param {string} libSearchTerm: The search term entered on the Library page
		 */
		function addResultMarkers(file, libSearchTerm) {

			// Open each file.
			var rawFile = new XMLHttpRequest();
			rawFile.open("GET", file, false);		
			rawFile.onreadystatechange = function() {
				if (rawFile.readyState == 4) {
					if (rawFile.status == 200 || rawFile.status == 0) {
						var allText = rawFile.responseText;

						// Each line contains a caption and a start time. Get each line.
						lines = allText.split('\n');

						/* Loop over each line. Split the line around the # symbol,
						to get the caption and the start time. */
						for (var i = 0; i < lines.length - 1; i++) {
							var line = lines[i].split('#');
							var caption = line[0];
							var startTime = line[1];

							// Remove stop words from the caption for the search.				
							var cleanCaption = removeStopWords(caption);

							/* If the cleaned caption contains the cleaned library search term, add the marker to the markers array.
							The marker will contain the original caption where stop words are not removed.
							This is because the caption sentence looks more natural with stop words. */
							if(cleanCaption.includes(libSearchTerm)) {								

								startTime = startTime.split(':')
								var startMin = parseInt(startTime[0]);
								var startSec = parseInt(startTime[1]);

								// Convert the start time to seconds. Round up, as this gives the marker a better position on the timeline.
								var startTimeOnSeekBar = Math.ceil((startMin * 60 + startSec));

								player.markers.add([{
									time: startTimeOnSeekBar,
									text: caption
								}]);

								// Add a button which, if clicked on, jumps to the current caption in the video.
								var captionButton = document.createElement("button");
								captionButton.setAttribute("id", "captionNavButton");
								captionButton.innerHTML = caption;
								captionButton.setAttribute("onclick", 'jumpToCaption("' + caption + '")');							
								document.getElementById("captionButtonsScroll").appendChild(captionButton);	
							}
						}
					}
				}
			}
			rawFile.send(null);
		}

		/**
		 * Reads a text file which contains captions and their start times.
		 * Puts the markers of all captions on the progress bar at their specified start time.
		 * @param {string} file: The text file to read
		 */
		function addAllMarkers(file) {

			// Open each file.
			var rawFile = new XMLHttpRequest();
			rawFile.open("GET", file, false);		
			rawFile.onreadystatechange = function() {
				if (rawFile.readyState == 4) {
					if (rawFile.status == 200 || rawFile.status == 0) {
						var allText = rawFile.responseText;

						// Each line contains a caption and a start time. Get each line.
						lines = allText.split('\n');

						/* Loop over each line. Split the line around the # symbol,
						to get the caption and the start time. */
						for (var i = 0; i < lines.length - 1; i++) {
							var line = lines[i].split('#');
							var caption = line[0];
							var startTime = line[1];

							startTime = startTime.split(':')
							var startMin = parseInt(startTime[0]);
							var startSec = parseInt(startTime[1]);

							// Convert the start time to seconds. Round up, as this gives the marker a better position on the timeline.
							var startTimeOnSeekBar = Math.ceil((startMin * 60 + startSec));

							player.markers.add([{
								time: startTimeOnSeekBar,
								text: caption
							}]);		

							// Add a button which, if clicked on, jumps to the current caption in the video.
							var captionButton = document.createElement("button");
							captionButton.setAttribute("id", "captionNavButton");
							captionButton.innerHTML = caption;
							captionButton.setAttribute("onclick", 'jumpToCaption("' + caption + '")');							
							document.getElementById("captionButtonsScroll").appendChild(captionButton);	
						}
					}
				}
			}
			rawFile.send(null);
		}

		/**
		 * Makes the video player jump to that part in the video 
		 * where the start time of the specified caption is.
		 * @param {string} caption: The caption to jump to in the video
		 */
		function jumpToCaption(caption) {

				// Get all markers of the video.
				var markers = player.markers.getMarkers();

				/* Find which marker contains the specified caption, 
				and jump to that marker */
				for (var i = 0; i < markers.length; i++) {

					var markerCaption = markers[i].text;		

					if (markerCaption.includes(caption)) {
						found = true;

						// Get the time of the marker, and jump there on the progress bar.
						var markerTime = markers[i].time;
						player.currentTime(markerTime);
						break;
					}
				}				
		}

		/**
		 * Takes the entered search term from the search box when the Search button is clicked.
		 * Looks for a marker that contains the search text.
		 * @param {string} searchTerm: The entered search term 
		 */
		function searchCaption(searchTerm) {	

			// Check if the user entered a search term.
			if(searchTerm.length > 0) {
				
				// Remove stop words from the search term.				
				var cleanSearchTerm = removeStopWords(searchTerm.toLowerCase());

				// Get all markers of the video.
				var markers = player.markers.getMarkers();

				/* Check if any marker's caption contains the search term. 
				If so, jump to that marker on the progress bar.
				If multiple such markers are found, jump to the first one. */
				var found = false;
				for (var i = 0; i < markers.length; i++) {

					var markerCaption = markers[i].text.toLowerCase();

					// Remove stop words from the marker's caption for the search.				
					var cleanMarkerCaption = removeStopWords(markerCaption);

					if (cleanMarkerCaption.includes(cleanSearchTerm)) {
						found = true;

						// Get the time of the marker, and jump there on the progress bar.
						var markerTime = markers[i].time;
						player.currentTime(markerTime);
						break;
					}
				}
				if (!found) {
					alert("No markers found.");
				}
			} else {
				alert("Please enter a search term.");
			}
		}

		// Show the playlist if the window is loaded.
		window.onload = function() {
			loadPlaylist();
		};
		
		 /**
		 * Adds buttons (video thumbnails) to the playlist using 
		 * the video names from the $playlistItems array.
		 */
		function loadPlaylist() {

			    // Get the names of the videos to be added to the playlist.
			    var playlistVideos = <?php echo json_encode($playlistItems); ?>;

				// Add each video from playlistVideos to the playlist.
			    for(var i = 0; i < playlistVideos.length; i++) {    	

			    	var videoName = playlistVideos[i];
 
			    	var playlistItem = document.createElement("input");
			    	playlistItem.setAttribute("type", "button");
			    	playlistItem.setAttribute("id", "playlistButton");
			    	playlistItem.setAttribute("value", videoName);

			    	// Show the saved video frame as a background image.
			    	playlistItem.setAttribute("style", "background-image:url(thumbnails/" + videoName + ".jpg);");

			    	var videoPath = "video_library/" + videoName + ".mp4";

			    	// Change the video if the thumbnail is clicked.
			    	playlistItem.setAttribute('onclick', 'changeVideo("'+videoPath+'")');

			    	// The video is added as a list item to the playlist.
			    	var listItem = document.createElement("li");
			    	listItem.appendChild(playlistItem);
			    	document.getElementById("playlist").appendChild(listItem);
			    }
		}

		/**
		* Changes the video being played to the clicked video of the playlist.
		* Also reads the caption file associated with the clicked video, so
		* the caption markers of that video will be displayed on the progress bar,
		* and the captions as buttons underneath.
		* @param {string} videoPath: The path to the clicked video
		*/
		function changeVideo(videoPath) {

			document.getElementById('test_video_html5_api').src = videoPath;	

			// Remove the markers of the previously played video.
			player.markers.removeAll();

			// Remove the caption buttons of the previously played video.
			var captionButtonsScroll = document.getElementById("captionButtonsScroll");
			while(captionButtonsScroll.firstChild) {
				captionButtonsScroll.removeChild(captionButtonsScroll.firstChild);
			}	

			// Dispaly the markers and captions of the new video.

			/* If the user entered a search term on the Library page, only add the markers of those
			captions of the video to be played which contain this term. */
			var libSearchTerm = "<?php echo $librarySearchTerm; ?>";

			if(libSearchTerm != null && libSearchTerm != "") {
				addResultMarkers(getCaptionFileOfCurrentVideo(), libSearchTerm);
			} 
			/* Otherwise, there was no search on the Library page, 
			so add all markers of the captions of the video to be played. */
			else {
				addAllMarkers(getCaptionFileOfCurrentVideo());
			}

			player.play();
		}

		/**
		* Returns the name of the caption file associated with the video currently loaded in the player.
		* It does not look in the file system, but simply constructs a string.
		* @return {string} The name of the caption file 
		*/
		function getCaptionFileOfCurrentVideo() {

			// Get the source of the video currently loaded in the player.
			var videoFile = document.getElementById('test_video_html5_api').src;

			// Split the path, and get the last element which is the file name.
			var pathElements = videoFile.split("/");
			var fileName = pathElements[pathElements.length - 1];

			// Remove ".mp4" from the end of the file name, and append ".txt" to it.
			var captionFile = "captions/" + fileName.substring(0, fileName.length - ".mp4".length) + ".txt";

			return captionFile;
		}
	</script>

	</body>

</html>