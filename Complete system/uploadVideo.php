<?php

/**
* Author: Albert Jozsa-Kiraly
* Project: Image/video auto-captioning using Deep Learning
* 
* This script places the browsed video file in the video library, 
* then runs the process_video.py Python script, and redirects to the status.html page.
*/
	
// Check if $_FILES['video_file'] is set and not null.
if(isset($_FILES['video_file'])) {
		
	$fileName = $_FILES['video_file']['name'];
	$fileTmpName = $_FILES['video_file']['tmp_name'];
	$fileExtension = strtolower(end(explode('.', $_FILES['video_file']['name'])));
				
	/* Replace each whitespace with an underscore in the name of the video.
	If any whitespace was left in the name, the thumbnail would not be displayed in the playlist. */
	$fileName = str_replace(' ', '_', $fileName);
	$fileTmpName = str_replace(' ', '_', $fileTmpName);		
		
	$validExtensions = array("mp4", "avi", "wmv", "flv");
		
	/* If the video file has a valid extension, upload it to the video library, 
	then run the processing script, and redirect to the status page. */
	if(in_array($fileExtension, $validExtensions) === true) {		
			
		$uploadPath = getcwd() . '/video_library/' . $fileName;			
		move_uploaded_file($fileTmpName, $uploadPath);			

		// Execute the Python script in the background without PHP waiting for it to finish.
		executeInBackground('py process_video.py');		
		
		// Redirect to the status page.
		header("Location: status.html");		
		exit();
			
	} else {
		echo "<script>alert('Please choose a valid video file.'); window.location.replace('home.html');</script> ";
	}	
}

/**
* Runs a specified command in the background.
* Uses a different method to run the command on a Windows system or a UNIX system.
*/
function executeInBackground($command) {
	if(substr(php_uname(), 0, 7) == "Windows") {
		pclose(popen("start /b ". $command, "r"));
	} 
	else {
		exec($command . " > /dev/null &");
	}
}
?>