# DSR-Bird-Song

## What is it?

This is the repo for our Portfolio project for [DSR](https://datascienceretreat.com/). Goal is to classify birds from their songs.

Contributors: 
* [Tim Bauer](https://github.com/bimtauer)
* [Satyan Sharma](https://github.com/stynshrm)
* [Falk Van der Meirsch](https://github.com/multavici)

## Data

The bird recordings are downloaded form the [Xeno-Canto database](https://www.xeno-canto.org/) with 
```
python data_preparation/script_download_files.py
```

## Pre-processing

To train the model the recordings first need to be converted in spectrograms. There are different ways of doing this:
* Fourier transformation: stft_s
* Mel spectrogram: mel_s
* Chirp spectrogram: chirp_s

Because our server space is limited, we choose to only make spectrograms from the parts in the recordings where a bird is actually present. Following the methodology described in [Sprengel et al., 2016](http://ceur-ws.org/Vol-1609/16090547.pdf), [this script](data_preparation/Signal_Extraction.py) localizes spectrogram sections with amplitudes above 3 times frequency- and time-axis medians, allowing us to extract audio sections most likely containing foreground bird vocalizations.

## Model

A convolutional neural network is used on the spectrograms to classify. We use the following Neural Networks:
* Bulbul [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* Sparrow [(Grill & Schlüter, 2017)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347092.pdf)
* SparrowExp [(Schlüter, 2018)](http://www.ofai.at/~jan.schlueter/pubs/2018_birdclef.pdf)

We propose two alternatives:
* 

## Model training

For model training we work with [Paperspace](https://www.paperspace.com/)

Running a job: 

```
paperspace jobs create --command "python import_test.py" --container "multavici/bird-song:latest" --apiKey <api-Key> --workspace "https://github.com/multavici/DSR-Bird-Song" --machineType "G1"
```
