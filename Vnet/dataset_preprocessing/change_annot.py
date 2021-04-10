import cv2
import os

annotpath = "/datahdd/WaterDetection/final_water_video_dataset/Test/annot/"#"/datahdd/dataset/water_segmentation/Train/annot222/"
savepath = "/datahdd/WaterDetection/final_water_video_dataset/Test/annot/"

imglist = sorted(os.listdir(annotpath))
print(imglist)

width = 9*16*4
height = 5*16*4

for i in imglist:
    if i[-4:] != '.png':
        continue
    img = cv2.imread(annotpath+i)
    print(img.shape)
    img[img == 3] = 0
    img[img == 7] = 125
    img[img == 35] = 255
    img = img[1:-1, 1:-1, :]
    print(img.shape)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    print(img.shape)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(savepath+i,img)
    print("save image ", i)