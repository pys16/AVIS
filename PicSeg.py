"""这个文件用来做图像的分割，即把一个图片分成两个部分。原理是两个视频结合的地方RGB值变化剧烈"""
import cv2
import os
import numpy as np

def div(channel):#divide the image into 2 parts
    delta=channel.copy()

    d=0
    for j in range(1, len(channel[0])):#列数 按列遍历
        for i in range(0, len(channel)):
            delta[i][j]=abs(int(delta[i][j]) - int(channel[i][j - 1]))#每一列减去前一列
    lin=[0]*len(channel[0])

    for j in range(1, len(channel[0])):
        for i in range(0, len(channel)):
            lin[j-1]=lin[j-1]+delta[i][j]
    d=lin.index(max(lin))
    return d

def Seg(img_path="test.jpg", out_path=None, out_name=None):
    """
    :param img_path:The Path to the image file
    :param out_path: The Output Path. If None, just return the div
    :param out_name: The Output Name.因为要用来做批处理所以加了个这个……
    :return:if out_path==None return d
    """
    im=cv2.imread(img_path)
    b,g,r=cv2.split(im)
    d1=div(b)
    d2=div(g)
    d3=div(r)
    # print(d1,d2,d3) # debug info
    if abs(d1-d2)<=3 or abs(d1-d3)<=3:
        d=d1
    elif abs(d2-d3)<=3:
        d=d2
    else:
        d=round(len(b[0])/2)#防止失效：直接取中间
        print("Pic_Seg Failed. Default to the midth")
    if out_path:
        crop_img1=im[0:len(b),0:d]
        crop_img2=im[0:len(b),(d+1):len(b[0])]
        if isinstance(out_name,int):
            cv2.imwrite(out_path + str(out_name)+'.jpg', crop_img1)
            cv2.imwrite(out_path+str(out_name+1)+'.jpg', crop_img2)
        else:
            cv2.imwrite(os.path.join(out_path,out_name+'p1.jpg'),crop_img1)
            cv2.imwrite(os.path.join(out_path,out_name+'p2.jpg'),crop_img2)
    else:
        return d

if __name__=="__main__":
    Seg(out_path='./',out_name=1)

