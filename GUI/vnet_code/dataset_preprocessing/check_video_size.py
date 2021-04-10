import cv2
import os

videopath = "/home/hyeongeun/dataset/Train/video/" #"/datahdd/dataset/water_segmentation/Test/video/"
videopath2 = "/home/hyeongeun/dataset/Test/video/" #"/datahdd/dataset/water_segmentation/Test/video/"

'''
print("Trainset video size")
videolist = sorted(os.listdir(videopath))

for video in videolist:
    totalpath = videopath + video

    vidcap = cv2.VideoCapture(totalpath)
    success, image = vidcap.read()
    if success:
        print("video name : ", video[:-4]," / video size : ", image.shape)
'''

print("="*50)
print("Testset video size")
videolist2 = sorted(os.listdir(videopath2))
hd_cnt = 0
rot_cnt = 0
fhd_cnt =0
n_cnt = 0
for video in videolist2:
    totalpath = videopath2 + video

    vidcap = cv2.VideoCapture(totalpath)
    success, image = vidcap.read()
    if success:
        print("video name : ", video[:-4]," / video size : ", image.shape)

    if image.shape[0]==720 :
        hd_cnt += 1
    elif image.shape[0]==1280 :
        rot_cnt += 1
    elif image.shape[0]==1080 :
        fhd_cnt += 1
    else :
        n_cnt += 1

print("720,1280 = {} , 1280,720 = {}".format(hd_cnt, rot_cnt))
print("1080,1920 = {} , no = {}".format(fhd_cnt, n_cnt))

