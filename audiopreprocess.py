import scipy.io.wavfile as wave
from python_speech_features import mfcc
import os
import cv2
import numpy as np
from pydub import audio_segment
import random

def process(audiopath='/run/media/pys/Data/Shiting/AudioSet/',segpath='/run/media/pys/Data/Shiting/Part/',imgpath='/run/media/pys/Data/Shiting/AudioImg/'):
    LabelDict={
        "accordion":0,
        "acoustic":1,
        "cello":2,
        "flute":3,
        "saxophone":4,
        "trumpet":5,
        "violin":6,
        "xylophone":7
    }
    # with open('label.txt','w') as f:
    sound=os.listdir(audiopath)

    for wav in sound:
        part=audio_segment.AudioSegment.from_wav(audiopath+wav)
        # part[0:4000].export(os.path.join(segpath,wav[:-4]+'p1.wav'),format='wav')
        index=random.randint(0,len(part)-4001)
        part[index:index+4000].export(os.path.join(segpath,wav[:-4]+'p2.wav'),format='wav')
    sound=os.listdir(segpath)
    for wav in sound:

        fs, data=wave.read(os.path.join(segpath+wav))
        if len(data.shape)==2:
            data = (data[:, 0] + data[:, 1]) / 2
        elif len(data.shape)>2:
            raise ValueError('Multichannel')
        features=mfcc(data,fs,nfft=4096)
        # print(features.shape)
        img=np.array(features)

        wav=wav[:-3]+'jpg'
        cv2.imwrite(imgpath + wav ,img)


if __name__=='__main__':

    process()
    