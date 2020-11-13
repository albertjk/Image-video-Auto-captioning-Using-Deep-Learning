<?php
/**
* Author: Albert Jozsa-Kiraly
* Project: Image/video auto-captioning using Deep Learning
* 
* This script gets the names of the text files in the 'captions' directory storing
* caption-time mappings for each video.
*/

$dir = 'captions/';

// Get the list of files in the 'captions' directory.
$filesInDir = scandir($dir);
$numOfFiles = sizeof($filesInDir);

$files = array();

/* The first two elements are '.' and '..' so skip these.
Start iterating from the third element. */
for($i = 2; $i < $numOfFiles; $i++)
   {
   	// Add each file name to the $files array.
     array_push($files, $filesInDir[$i]);
   }

   /* Return the array as a JSON representation.
   This will be used by the JavaScript in player.html. */
   echo json_encode($files, JSON_PRETTY_PRINT);  
?>