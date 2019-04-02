"""
Based on Nussl
"""

import nussl
import wave
import os
import numpy as np

def audio_decomp(video=None,audio=None,OutputPath=None,file=None):
    signal=nussl.AudioSignal(audio_data_array=audio[0,:],sample_rate=44100)
    nmf_mfcc=nussl.NMF_MFCC(signal,num_sources=2, num_templates=25, distance_measure="euclidean", random_seed=0)
    nmf_mfcc.run()
    sources=nmf_mfcc.make_audio_signals()
    source_list=[]
    for i,source in enumerate(sources):
        outputfile=os.path.join(OutputPath,file+'__seg'+str(i+1)+'.wav')
        source.write_audio_to_file(outputfile)
        source_list.append(source.audio_data)

    return [os.path.join(OutputPath,file+"__seg1.wav"),os.path.join(OutputPath,file+"__seg2.wav")],source_list

def read_audio(audio_file):
    f = wave.open(audio_file,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    waveData = np.fromstring(strData,dtype=np.int16)
    # normalize
    waveData = waveData*1.0/(max(abs(waveData)))
    time = np.arange(0,nframes)*(1.0 / framerate)
    return waveData

def main():
    # f = wave.open('accordion_2_acoustic_guitar_2.wav', 'rb')
    # params = f.getparams()
    # nchannels, sampwidth, framerate, nframes = params[:4]
    # strData = f.readframes(nframes)
    # waveData = np.fromstring(strData, dtype=np.int16)
    # # normalize
    # waveData = waveData * 1.0 / (max(abs(waveData)))
    # time = np.arange(0, nframes) * (1.0 / framerate)
    # audios=[]
    # audios.append(waveData)
    # npaudios=np.array(audios)
    # audio_decomp(audio=npaudios,OutputPath='./',file='accordion_2_acoustic_guitar_2')
    file=os.listdir('/run/media/pys/Data/Shiting/testset25/gt_audio')
    if not os.path.exists('./AudioSeg/'):
            os.mkdir('./AudioSeg')
    for i in file:
        wavedata=read_audio('/run/media/pys/Data/Shiting/testset25/gt_audio/'+i)
        audios=[]
        audios.append(wavedata)
        npaudios=np.array(audios)

        audio_decomp(audio=npaudios,OutputPath='./AudioSeg/',file=i[:-4])




if __name__=='__main__':
    main()