import os
import random
import re
import json
import time

import tensorflow as tf
import numpy as np

import run
import audiode
import PicSeg
import audiopreprocess

"""
Image Recognition
"""
#分割
def ImgSeg(ImgPath,FileName):
    SetName=ImgPath.split('/')[-1]
    if not os.path.exists('./Working/ImgSeg'):
        os.mkdir('./Working/ImgSeg')
    if not os.path.exists(os.path.join('./Working/ImgSeg',SetName)):
        os.mkdir(os.path.join('./Working/ImgSeg',SetName))
    PicSeg.Seg(img_path=os.path.join(ImgPath,FileName),out_path=os.path.join('./Working/ImgSeg/',str(SetName)),out_name='')#File name contains .jpg suffix
#识别
def ImgRec(Val_info):
    run.main(1)
    Key=[]
    Value=[]
    with open('preds.txt','r') as infile:
        lines=infile.readlines()
        for line in lines:
            k=line.split('\'')[1]#引号内部是文件名
            v=re.split(',\s',line)[2]
            if k.split('/')[-2]!='Exception':
                Key.append(k)
                Value.append(v)

    return Key,Value
"""
AudioDecomposition
"""
def AudioDecomp(debug=False):

    file=os.listdir('./gt_audio')
    if not os.path.exists('./Working/AudioSeg'):
        os.mkdir('./Working/AudioSeg')
    for i in file:
        if debug and os.path.exists('./result_audio/'+i[:-4]+'__seg1.wav'):#仅debug时使用：跳过分解（真的太慢了）
            os.system('cp ./result_audio/' + i[:-4] + '__seg1.wav ./Working/AudioSeg/')
            os.system('cp ./result_audio/' + i[:-4] + '__seg2.wav ./Working/AudioSeg/')
        else:
            wavedata = audiode.read_audio('./gt_audio/' + i)
            audios = []
            audios.append(wavedata)
            npaudios = np.array(audios)

            audiode.audio_decomp(audio=npaudios, OutputPath='./Working/AudioSeg/', file=i[:-4])
"""
AudioRecognition
"""
#音源识别
def AudioRec(SetNum):
    if not os.path.exists('./Working/AuPiece'):
        os.mkdir('./Working/AuPiece')
    if not os.path.exists('./Working/AuImg'):
        os.mkdir('./Working/AuImg')
    audiopreprocess.process(audiopath='./Working/AudioSeg/',segpath='./Working/AuPiece/',imgpath='./Working/AuImg/')
    Key=[]
    Value=[]
    file=os.listdir('./Working/AuImg')
    with open('validsound.txt','w') as vs:
        for i in file:
            vs.write(os.path.join('./Working/AuImg',i)+'\n')
        if (SetNum*2)%16!=1:
            for i in os.listdir('./ExcptSnd'):
                vs.write('./ExcptSnd/'+i+'\n')
    tf.reset_default_graph()
    run.main(2)
    with open('predsound.txt','r') as infile:
        lines=infile.readlines()
        for line in lines:
            k = line.split('\'')[1]  # 引号内部是文件名
            v = re.split(',\s', line)[2]
            if k.split('/')[-2]!='ExcptSnd':
                Key.append(k)
                Value.append(v)

    return Key, Value


def main():
    LabelDict={
        '0':"accordion",
        '1':"acoustic",
        '2':"cello",
        '3':"flute",
        '4':"saxophone",
        '5':"trumpet",
        '6':"violin",
        '7':"xylophone"
    }
    SPath=os.listdir('./testimage/')#list the ImagePath
    SetNum=len(SPath)#测试集的数目
    if not os.path.exists('./Working'):
        os.mkdir('./Working')
    print('======Work Begins======')
    time0=time.time()
    print('※Image Segmentation')
    with open('validation.txt','w') as file:
        #Step1: 分割图片并写入磁盘
        SPath=os.listdir('./testimage/')#list the ImagePath
        SetNum=len(SPath)#测试集的数目
        for imgpath in SPath:
            ImgFile=os.listdir('./testimage/'+imgpath)
            Index=random.randint(int(len(ImgFile)/3.),int(len(ImgFile)/3.*2))#Capture an image randomly
            ImgSeg('./testimage/'+imgpath,ImgFile[Index])#Corp and save
        #Step2：将图片目录写到validation.txt
        SetPath=[]
        ImgPath=[]
        for set in os.listdir('./Working/ImgSeg'):#listdir集合名
            SetPath.append(os.path.join('./Working/ImgSeg/',set))
        for setpath in SetPath:
            for imgpath in os.listdir(setpath):
                ImgPath.append(os.path.join(setpath,imgpath))
        for img in ImgPath:
            file.write(img+'\n')#所有图片写入validation.txt
        if (SetNum*2)%8!=1:#填满batch
            for i in os.listdir('./Exception'):
                file.write('./Exception/'+i+'\n')
    time1=time.time()
    print('time:',time1-time0)
    print('※Image Recognition')
    ImKey,ImValue=ImgRec('./Working/ImgSeg/')
    time2=time.time()
    # print('time:',time2-time1)
    print('※Audio Decomposition(Very Slow!)')
    #AudioDecomp有可能跳过！
    AudioDecomp(True)
    time3=time.time()
    print('time:',time3-time2)
    print('※Audio Recognition')
    AuKey,AuValue=AudioRec(SetNum)
    time4=time.time()
    print('time:',time4-time3)
    print('※Analysing datas...')
    print(ImKey,ImValue,AuKey,AuValue)
    Result=[]
    #Result=['视频名','左边乐器编号','右边乐器编号','left/right','乐器编号','right/left','乐器编号']
    for imgpath in SPath:
        Result.append([imgpath])
    for index in range(0,SetNum):
        #以图像结果为真值，每一个图像分左右两个部分
        #一轮，不管左右append即可
        for ik in range(0,len(ImKey)):
            if ImKey[ik].split('/')[-2]==Result[index][0]:
                Result[index].append(ImValue[ik])
                ImKey.pop(ik)
                ImValue.pop(ik)#pop的原因是防止二轮重复检测
                break
        #二轮，需要判断左右
        for ik in range(0,len(ImKey)):
            if ImKey[ik].split('/')[-2]==Result[index][0]:
                if ImKey[ik][-5]=='1':#它自己是左，insert
                    Result[index].insert(1,ImValue[ik])
                else:#它自己是右，append
                    Result[index].append(ImValue[ik])
                ImKey.pop(ik)
                ImValue.pop(ik)
                break
        #音频识别结果，有可能没有结果
        for ia in range(0,len(AuKey)):#AuKey和AuValue下标
            Filename=AuKey[ia].split('/')[-1]
            if Filename.split('__')[0]==Result[index][0]:
                PartName=Filename[-7]
                if AuValue[ia]==Result[index][1]:
                    Result[index].append('Left')
                    Result[index].append(PartName)
                elif AuValue[ia]==Result[index][2]:
                    Result[index].append('Right')
                    Result[index].append(PartName)
                # AuKey.pop(ia)
                # AuValue.pop(ia)
        if len(Result[index])>3:#如果做出了结果
            if Result[index][3]=='Left':
                Result[index][3]=Result[index][4]
                Result[index][4]=str(3-int(Result[index][4]))
            else:
                Result[index][3] = str(3-int(Result[index][4]))
        else:
            Result[index].append('1')
            Result[index].append('2')
    OutPut={}
    Key1='0'
    Key2='1'
    Key3='position'
    Key4='audio'
    print(Result)
    for Line in Result:
        if Line[3]=='1':
            Key0=Line[0]+'.mp4'
            OutPut[Key0]=[
                {
                    Key3: 0,
                    Key4:  Line[0] + '__seg' + Line[3] + '.wav'
                },
                {
                    Key3: 1,
                    Key4:  Line[0] + '__seg' + Line[4] + '.wav'
                }
            ]
        else:
            Key0=Line[0]+'.mp4'
            OutPut[Key0]=[
                {
                    Key3: 1,
                    Key4:  Line[0] + '__seg' + Line[3] + '.wav'
                },
                {
                    Key3: 0,
                    Key4:  Line[0] + '__seg' + Line[4] + '.wav'
                }
            ]
    if not os.path.exists('./result_json'):
        os.mkdir('./result_json')
    with open('./result_json/result.json','w') as f:
        outjson=json.dump(OutPut,f)
    if not os.path.exists('./result_audio/'):
        os.mkdir('./result_audio')
    os.system('cp -rf ./Working/AudioSeg/. ./result_audio/')#音频分离输出写错了所以复制一下……
    # os.system('rm -rf ./Working/')#这一行会删除Working这样就能连续跑了
    time5=time.time()
    print('time:',time5-time4)

if __name__=='__main__':
    main()