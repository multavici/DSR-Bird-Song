# -*- coding: utf-8 -*-
import librosa

def resample(ipath, opath):
    try:
        audio, sr = librosa.load(ipath, sr = 22050, mono = True)
        librosa.output.write_wav(opath, audio, sr, norm = [-1, +1])
    except:
        print(f"Error with file {ipath}")
    
    
    
"""   
i = 'Test Birdsounds/anser%20brachyrhynchus0.mp3'
o = 'test.wav'

resample(i, o)
""" 