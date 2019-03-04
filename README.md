# DSR-Bird-Song  

This is the repo for our Portfolio project for [DSR](https://datascienceretreat.com/). Goal is to classify birds from their songs.

Contributors: 
* [Tim Bauer](https://github.com/bimtauer)
* [Satyan Sharma](https://github.com/stynshrm)
* [Falk Van der Meirsch](https://github.com/multavici)

For model training we work with [Paperspace](https://www.paperspace.com/)

Running a job: 

```
paperspace jobs create --command "python import_test.py" --container "multavici/bird-song:latest" --apiKey <api-Key> --workspace "https://github.com/multavici/DSR-Bird-Song" --machineType "G1"
```

The bird recordings are downloaded form the [Xeno-Canto database](https://www.xeno-canto.org/) with 
```
python data_preparation/script_download_files.py
```


To train the model the recordings first need to be converted in spectrograms. There are different ways of doing this:
* Fourier transformation: stft_s
* Mel spectrogram: mel_s
* Chirp spectrogram: chirp_s

Because our server space is limited, we choose to only make spectrograms from the parts in the recordings where a bird is actually present. [This script](data_preparation/Signal_Extraction.py) localizes the parts where the volume is 

On these spectrograms a convolutional neural network is used to classify. We use the following Neural Networks:
* Bulbul
* Sparrow
* ..

We propose two alternatives:
* 