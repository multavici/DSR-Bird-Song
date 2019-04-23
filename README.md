# DSR-Bird-Song

## Live demo

[here](https://chirps.eu)

## What is it?

This is the repo for our Portfolio project for [DSR](https://datascienceretreat.com/). The goal is to train a model that can reliably classify bird species from their songs and make it available as a webservice/app. 
Our motivation is twofold: we want to contribute to the development of tools for automated biodiversity monitoring and provide bird enthusiasts with a handy tool.

Contributors: 
* [Tim Bauer](https://github.com/bimtauer)
* [Satyan Sharma](https://github.com/stynshrm)
* [Falk Van der Meirsch](https://github.com/multavici)

## Data

The bird recordings are downloaded form the [Xeno-Canto database](https://www.xeno-canto.org/) with 
```
python data_preparation/audio_acquisition/download_files_threaded.py
```

## Pre-processing

The audio recordings vary greatly in quality and number of species present. Assuming that the foreground species is usually the loudest in a recording we follow the methodology described in [Sprengel et al., 2016](http://ceur-ws.org/Vol-1609/16090547.pdf) to extract signal sections from a noisy background. [This script](birdsong/data_preparation/audio_conversion/signal_extraction.py)  localizes spectrogram sections with amplitudes above 3 times frequency- and time-axis medians, allowing us to extract audio sections most likely containing foreground bird vocalizations. We run the script over all recordings in our storage and store the respective timestamps for signal sections in our database.

To train the model the recordings first need to be converted in spectrograms. There are different ways of doing this:
* Fourier transformation: stft_s
* Mel spectrogram: mel_s
* Chirp spectrogram: chirp_s

A key-challenge in pre-processing is that we want to maintain flexibility in terms of spectrogram functions and parameters. Storage space is limited and the amount of available recordings vast. Thus we developed a [custom implementation](birdsong/datasets/dynamic_dataset.py) of the PyTorch Dataset class that makes use of a background process to dynamically load and convert audio. Ideally this process should be fast enough to preload an entire batch during training time. But a major bottleneck for audio-loading is resampling. Thus we chose to resample all files in our database to 22050hz as a first step in order to be able to load them with native sample rate later on.

For rapid model development we are currently using a small subsample of the data for which we have precomputed mel-spectrograms. 

## Model

We build the following [models](birdsong/models):
* [Bulbul](birdsong/models/bulbul.py): [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* [Sparrow](birdsong/models/sparrow.py): [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* [SparrowExp](birdsong/models/sparrow_exp_a.py): [(Schlüter, 2018)](http://www.ofai.at/~jan.schlueter/pubs/2018_birdclef.pdf)
* [Zipzalp](birdsong/models/zilpzalp.py): own creation by Tim 


## Model training
Configure your run in scripts/config.py


Running a job locally: 
```
sh run.sh
```
Running a job on [Paperspace](https://www.paperspace.com/): 
```
paperspace jobs create --command "sh run.sh" --container "multavici/bird-song:latest" --apiKey <api-Key> --workspace "https://github.com/multavici/DSR-Bird-Song" --machineType "G1"
```

Enter bash in docker container with current PWD mounted:
```
docker run -it --mount src="$(pwd)",target=/test,type=bind multavici/bird-song /bin/bash
```
