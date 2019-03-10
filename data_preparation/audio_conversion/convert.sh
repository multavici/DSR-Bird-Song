#!/bin/bash

for i in storage/top10_german_birds_converted/*.mp3;

do ffmpeg -i ${i} -b:a 128k -ar 22050 -ac 1 -y  ${i}; 

done