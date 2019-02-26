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
````

