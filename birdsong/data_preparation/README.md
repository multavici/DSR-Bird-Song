# Data preparation

The main source of our data is the Xeno-Canto library. [Xeno-Canto](https://www.xeno-canto.org/) is a website dedicated to sharing bird sounds from all over the world. They were so kind to let us have free access to their database.

## Getting all the species

First we [scraped](/get_taxonomy_xeno-canto.py) the Xeno-Canto website to get the taxonomy of all the species found on the website and saved it in a [json file](/taxonomy.txt).

## Creating a sqlite database

Secondly we created a lightweight SQLite database with two tables: 

* taxonomy
* recordings

The taxonomy table contains all the bird species (Order, Family, Genus, Species). We inserted the species extracted in the first step in this table. The recordings table will contain the metadata of all the recordings found in the database.

## Getting the recordings metadata

This [script](https://www.xeno-canto.org/api/2/recordings?query=foo) uses the [Xeno-Canto api](https://www.xeno-canto.org/api/2/recordings?query=foo) to get the metadata of all the recordings in their database and save it to our SQLite database.

At this point we decided to focus on the bird species that are found in Germany.

We [scraped](/german_bird_list.py) the [wikipedia page](https://commons.wikimedia.org/wiki/Liste_der_V%C3%B6gel_Deutschlands) for german species and saved it as a boolean field in the taxonomy database table.

## Downloading the recordings

This [script](/script_download_files_threaded.py) downloads all mp3 files from german species to the storage location. We use threads to drastically speed up this process.

Once downloaded, we flagged all the downloaded recordings in the database.

## Getting extra metadata

Some of the metadate of the recordings (mostly technical audio data) is not included in the info that we get through the Xeno-Canto api but is available on the [webpage](https://www.xeno-canto.org/12345) of each recording. 
This script gets the metadata that we found useful: duration, sample rate, bitrate and amount of channels (1 for mono, 2 for stereo)

## Converting the audio

The recordings from Xeno-Canto are all mp3 files but come in a myriad of different bitrates, sample rates and channels. To make the pre-processing step while training the model as fast as possible we chose to convert all the recordings to a same set of output paramaters:

* sample rate: 22050 /s
* bitrate: 128 kb/s
* channels: 1 (mono)

We used [ffmpeg](http://ffmpeg.org/) on the paperspace server in [this](https://hub.docker.com/r/lansmash/docker-ffmpeg) docker container.