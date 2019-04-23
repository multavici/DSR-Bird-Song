# DSR-Bird-Song

## What is it?

This is the repo for our Portfolio project for [DSR](https://datascienceretreat.com/). The goal is to train a model that can reliably classify bird species from their songs and make it available as a webservice/app. 
Our motivation is twofold: we want to contribute to the development of tools for automated biodiversity monitoring and provide bird enthusiasts with a handy tool.

Contributors: 
* [Tim Bauer](https://github.com/bimtauer)
* [Satyan Sharma](https://github.com/stynshrm)
* [Falk Van der Meirsch](https://github.com/multavici)

## Live demo: [chirps.eu](https://chirps.eu)

Start page             |  Classification
:-------------------------:|:-------------------------:
[![screenshot](https://github.com/multavici/DSR-Bird-Song/blob/master/app/static/images/app_screenshot.png?raw=true)](https://chirps.eu)  |  [![screenshot](https://github.com/multavici/DSR-Bird-Song/blob/master/app/static/images/screenshot_app_classified.png?raw=true)](https://chirps.eu)

## Data

The bird recordings are downloaded form the [Xeno-Canto database](https://www.xeno-canto.org/) with 
```
python data_preparation/audio_acquisition/download_files_threaded.py
```

## Pre-processing

The audio recordings vary greatly in quality and number of species present. Assuming that the foreground species is usually the loudest in a recording we follow the methodology described in [Sprengel et al., 2016](http://ceur-ws.org/Vol-1609/16090547.pdf) to extract signal sections from a noisy background. [This script](birdsong/data_preparation/audio_conversion/signal_extraction.py)  localizes spectrogram sections with amplitudes above 3 times frequency- and time-axis medians, allowing us to extract audio sections most likely containing foreground bird vocalizations. We run the script over all recordings in our storage and store the respective timestamps for signal sections in our database.

Initially our aim was to store only raw audio and integrate the preprocessing of
spectrograms into a [custom PyTorch Dataset](birdsong/datasets/dynamic_dataset.py). That way we would have retained flexibility in terms of the spectrogram functions and parameters. But despite extensive efforts in cutting down preprocessing time, data loading remained the main bottleneck in out training times.

Thus the decision was made to precompute spectrogram slices according to a procedure common in the literature: 5 second slices with 2.5 second overlap where first converted into spectrograms using STFT (FFT window size: 2048, hop length: 512) and then passed through a 256 Mel filterbank resulting in Mel-Spectrograms with a dimension of 
256 x 216 x 1 as input to our models.

For our early approaches we rebuild models we found in the literature which all 
treat the audio classification task as an image problem. Through experiments with
non-square kernels we tried to take into account the different nature of information on the time axis versus the frequency axis of the spectrogram, leading to improved results (Hawk, Zilpzalp). A breakthrough came when we decided to couple a CNN with an LSTM, designing the former to pick up on timbral features and the latter to detect time-dependent patterns (Puffin).

## Model

We build the following [models](birdsong/models):
* [Bulbul](birdsong/models/bulbul.py): [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* [Sparrow](birdsong/models/sparrow.py): [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* [SparrowExp](birdsong/models/sparrow_exp_a.py): [(Schlüter, 2018)](http://www.ofai.at/~jan.schlueter/pubs/2018_birdclef.pdf)
* [Zipzalp](birdsong/models/zilpzalp.py): own creation by Tim 
* [Hawk](birdsong/models/hawk.py): [(Pons et al., 2018)](http://ismir2018.ircam.fr/doc/pdfs/191_Paper.pdf )
* [Puffin](birdsong/models/puffin.py): own creation by Tim 


## Model training
Configure your run in scripts/config.py

Running a job locally: 
```
python train_precomputed.py
