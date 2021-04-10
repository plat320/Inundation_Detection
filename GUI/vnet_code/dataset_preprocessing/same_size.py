import cv2
import os

gtpath = "/datahdd/dataset/water_segmentation/Test/annot/"
frmpath = "/datahdd/dataset/water_segmentation/Test/frames/"

frmlist = sorted(os.listdir(frmpath))
gtlist = sorted(os.listdir(gtpath))

print(len(frmlist))
print(len(gtlist))

if len(frmlist)==len(gtlist):
    for i in range(len(frmlist)):
        frm = cv2.imread(frmpath+frmlist[i]+"/frm000000.jpg")
        gt = cv2.imread(gtpath+gtlist[i])
        if frm.shape == gt.shape:
            print(gtlist[i], "already ok")
        else :
            gt = cv2.resize(gt,(frm.shape[1],frm.shape[0]),cv2.INTER_NEAREST)
            cv2.imwrite(gtpath+gtlist[i],gt)
            print(gtlist[i], " changed")

else :
    print("different length")