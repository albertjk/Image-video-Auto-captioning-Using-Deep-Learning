<?php	
/**
* Author: Albert Jozsa-Kiraly
* Project: Image/video auto-captioning using Deep Learning
* 
* This script gets the name of the most recent video added to the video library, 
* and continuously checks if the associated caption text file exists.
*/

/**
* This function gets the name of the most recent video in the library.
* This is the video that was just uploaded.
* @param string $libraryDir: Path to the video library directory
* @return string: Path to the most recent video
*/
function getMostRecentVideo($libraryDir) {
	
	$max = ['path' => null, 'timestamp' => 0];
	
	foreach (scandir($libraryDir, SCANDIR_SORT_NONE) as $file) {
		$path = $libraryDir . '/' . $file;
		
		if(!is_file($path)) {
			continue;
		}
		
		$timestamp = filemtime($path);
		if($timestamp > $max['timestamp']) {
			$max['path'] = $path;
			$max['timestamp'] = $timestamp;
		}
	}
	
	return $max['path'];	
}

$libDir = "video_library";
$fileName = getMostRecentVideo($libDir);

// Remove the path and only keep the video name.
$fileName = explode("/", $fileName)[1];

// This is the name of the caption file to check for if it has been created already.
$captionFileName = "captions/" . explode(".", $fileName)[0] .".txt";

// Continuously check if the caption file has been created.
while(true) {
	if(file_exists($captionFileName)) {
		echo "Caption file exists.";
		break;
	}	
}			
?>