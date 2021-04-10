import cv2
import os

videopath = "/datahdd/WaterDetection/final_water_video_dataset/Train/video/" #"/datahdd/dataset/water_segmentation/Test/video/"
savepath = "/datahdd/WaterDetection/final_water_video_dataset/2d_dataset/Train/image/"#"/datahdd/dataset/water_segmentation/Test/frames/"

videolist = sorted(os.listdir(videopath))
print(videolist)

width = 9*16*4
height = 5*16*4

## video to frames
'''
for video in videolist:
    totalpath = videopath + video
    print("video path : ", totalpath)
    print("video name : ", video[:-4])
    dirpath = savepath+video[:-4]
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    vidcap = cv2.VideoCapture(totalpath)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        image = cv2.resize(image, (width, height))
        cv2.imwrite(dirpath+"/frm%06d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        if count>100:
            break
'''


## video to 2D dataset
for video in videolist:
    totalpath = videopath + video
    print("video path : ", totalpath)
    print("video name : ", video[:-4])
    dirpath = savepath
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    vidcap = cv2.VideoCapture(totalpath)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        image = cv2.resize(image, (width, height))
        cv2.imwrite(dirpath + video[:-4]+".jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        if count>0:
            break