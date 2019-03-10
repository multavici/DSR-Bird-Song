#!/bin/bash

# This script converts all mp3 files in the specified folder to files with
# bitrate 128k, sample rate 22050 and 1 channel (mono)

for i in /storage/all_german_birds/*.mp3;

do ffmpeg -i ${i} -b:a 128k -ar 22050 -ac 1 -y  ${i}; 

done