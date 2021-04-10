import cv2
import os

annotpath = "/datahdd/WaterDetection/each_dataset/Cheong-gye-cheon data/test/cgc_test_verti/annot/"
videopath = "/datahdd/WaterDetection/each_dataset/Cheong-gye-cheon data/test/cgc_test_verti/video/"
save_annotpath = "/datahdd/WaterDetection/each_dataset/Cheong-gye-cheon data/test/cgc_test_verti/annot_updown/"
save_videopath = "/datahdd/WaterDetection/each_dataset/Cheong-gye-cheon data/test/cgc_test_verti/video_updown/"

annotlist = sorted(os.listdir(annotpath))
videolist = sorted(os.listdir(videopath))
#print(annotlist)
width = 9*16*4
height = 5*16*4

for i in annotlist:
    if i[-4:] != '.png':
        continue
    annot = cv2.imread(annotpath + i)

    annot[annot == 3] = 0
    annot[annot == 7] = 125
    annot[annot == 35] = 255
    annot = annot[1:-1, 1:-1, :]
    annot = cv2.copyMakeBorder(annot, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    hh = annot.shape[0]
    ww = annot.shape[1]

    annot_up = annot[:hh//2, :, :]
    annot_down = annot[hh//2:, :, :]

    annot_up = cv2.resize(annot_up, (width, height), interpolation=cv2.INTER_NEAREST)
    annot_down = cv2.resize(annot_down, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_annotpath + i[:-4] + "_up.png", annot_up)
    cv2.imwrite(save_annotpath + i[:-4] + "_down.png", annot_down)
    print("save image ", i, " up,down")

for j in videolist:
    cap = cv2.VideoCapture(videopath+j)
    # Define the codec and create VideoWriter object
    fourcc_up = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc_down = cv2.VideoWriter_fourcc(*'MP4V')
    out_up = cv2.VideoWriter(save_videopath + j[:-4] + "_up.mp4", fourcc_up, 30.0, (width, height))
    out_down = cv2.VideoWriter(save_videopath + j[:-4] + "_down.mp4", fourcc_down, 30.0, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            hh = frame.shape[0]
            ww = frame.shape[1]

            frame_up = frame[:hh // 2, :, :]
            frame_down = frame[hh // 2:, :, :]

            frame_up = cv2.resize(frame_up, (width, height))
            frame_down = cv2.resize(frame_down, (width, height))

            # write the flipped frame
            out_up.write(frame_up)
            out_down.write(frame_down)
        else:
            print('save video ',j,'up, down')
            break

    # Release everything if job is finished
    cap.release()
    out_up.release()
    out_down.release()