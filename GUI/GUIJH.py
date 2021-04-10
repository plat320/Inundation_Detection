import sys
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import PIL.Image as Image
import PyQt5
import os
import test
import numpy as np
import cv2
from pathlib import Path
sys.path.append(os.path.abspath(".") + "/segmentation_models_pytorch/unet")


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.frm_num = 16



    def setupUI(self):
        self.setGeometry(50, 50, 1350, 500)

        self.timeVar = QTimer()
        self.timeVar.setInterval(200)
        self.timeVar2 = QTimer()
        self.timeVar2.setInterval(2147483647)
        self.num = 0


        self.reset_flag = 0
        self.set_flag = 0

        self.warn_label = QLabel(" ", self)
        self.warn_label.move(10, 40)
        self.warn_label.resize(300,30)


        self.result_label1 = QLabel(" ", self)
        self.result_label1.move(150, 400)
        self.result_label1.resize(200,20)
        self.result_label2 = QLabel(" ", self)
        self.result_label2.move(150, 420)
        self.result_label2.resize(200,20)
        self.result_label3 = QLabel(" ", self)
        self.result_label3.move(150, 440)
        self.result_label3.resize(200,20)


        subjectLabel = QLabel("Inundation Alert System", self)
        subjectLabel.move(10, 10)
        subjectfont = PyQt5.QtGui.QFont("", 14)
        subjectfont.setBold(True)
        subjectLabel.setFont(subjectfont)
        subjectLabel.resize(300, 40)

        subjectLabel1 = QLabel("MMI Lab  Team Where's water now?", self)
        subjectLabel1.move(1100, 410)
        subjectLabel1.resize(300, 40)

        self.label1 = QLabel(self)              # left image
        self.label1.move(150, 70)
        self.label1.resize(576, 320)

        self.label2 = QLabel(self)              # right result
        self.label2.move(750, 70)
        self.label2.resize(576, 320)


        btn2 = QPushButton("Run", self)
        btn2.move(20, 70)
        btn2.clicked.connect(self.btn2_clicked)
        btn2.resize(100, 30)


        btn3 = QPushButton("Reset", self)
        btn3.move(20, 120)
        btn3.clicked.connect(self.btn3_clicked)
        btn3.resize(100, 30)
        lb = QLabel()

        subjectLabel = QLabel("Choose image", self)
        subjectLabel.move(20, 160)
        subjectLabel.resize(150, 30)


        subjectLabel = QLabel("Choose model", self)
        subjectLabel.move(20, 230)
        subjectLabel.resize(150, 30)

        self.img_init()

        self.cmb_init()

    def btn2_clicked(self):             # Run
        if self.ComboBX.currentIndex() == 0 or self.ComboBX2.currentIndex() == 0:
            self.warn_label.setText("Please choose model and image")
            self.warn_label.setObjectName('Inundated')
            self.warn_label.setStyleSheet('QLabel#Inundated {color: black}')
            warnfont1 = PyQt5.QtGui.QFont("", 8)
            warnfont1.setBold(False)
            self.warn_label.setFont(warnfont1)
        else:
            WaterIoU, RoadIoU, inundation_rate, base_rate = test.create_img(self.ComboBX.currentText(), self.ComboBX2.currentText())
            self.timer2()
            self.img_label1.setText("reference + result")
            self.img_label2.setText("target + result")

            self.result_label1.setText("Water IoU = %.3f" %WaterIoU)
            self.result_label2.setText("Road IoU = %.3f" %RoadIoU)
            if inundation_rate == -1:
                self.result_label3.setText("This CCTV image has no road region" )     # inundation term 추가할 것 그러기위해서 reference image에서 road영역과 비교하는 코드 구현해야함
                self.result_label3.setText("This CCTV image has no road region" )     # inundation term 추가할 것 그러기위해서 reference image에서 road영역과 비교하는 코드 구현해야함
            else:
                self.result_label3.setText("Inundation score = %.3f"%(inundation_rate/base_rate))
            if inundation_rate>base_rate:
                self.warn_label.setText("This area has been inundated.")
                self.warn_label.setObjectName('Inundated')
                self.warn_label.setStyleSheet('QLabel#Inundated {color: red}')
                warnfont = PyQt5.QtGui.QFont("", 10)
                warnfont.setBold(True)
                self.warn_label.setFont(warnfont)


            if self.ComboBX2.currentIndex() <= 3:
                pixmap1 = QPixmap('./result/' + "ref_" + self.ComboBX.currentText() + ".png")
                self.label1.setPixmap(pixmap1)
                pixmap2 = QPixmap('./result/'+self.ComboBX.currentText()+".png")
                self.label2.setPixmap(pixmap2)
            else:
                self.timeVar.timeout.connect(self.timer)
                self.timeVar.start()
                self.timeVar2.timeout.connect(self.timer2)
                self.timeVar2.start()

    def timer(self):
        pixmap1 = QPixmap('./result/' + "ref_"+self.ComboBX.currentText()+ str(self.num % 16).zfill(2) + ".png")
        self.label1.setPixmap(pixmap1)
        pixmap2 = QPixmap('./result/' + self.ComboBX.currentText() + str(self.num % 16).zfill(2) + ".png")
        self.label2.setPixmap(pixmap2)
        self.num = self.num + 1

    def timer2(self):
        self.timeVar.stop()
        self.timeVar2.stop()


    def btn3_clicked(self):             # Reset
        # self.label1.clear()
        # self.label2.clear()
        pixmap1 = QPixmap('./white_mask.png')
        self.label1.setPixmap(pixmap1)
        pixmap2 = QPixmap('./white_mask.png')
        self.ComboBX.setCurrentIndex(0)
        self.ComboBX2.setCurrentIndex(0)
        self.warn_label.setText(" ")
        self.result_label1.setText(" ")
        self.result_label2.setText(" ")
        self.result_label3.setText(" ")
        self.set_flag = 0
        self.timer2()

    def cmb(self):                              # figure combobox
        if self.ComboBX.currentText() == " " or self.ComboBX.currentText() == "":
            self.timer2()
            pixmap1 = QPixmap('./white_mask.png')
            self.label1.setPixmap(pixmap1)
            pixmap2 = QPixmap('./white_mask.png')
            self.label2.setPixmap(pixmap2)
            self.img_label1.setText(" ")
            self.img_label2.setText(" ")
        else:
            self.timer2()
            pixmap1 = QPixmap('./flood/' + self.ComboBX.currentText() + '.jpg')
            self.label2.setPixmap(pixmap1)
            pixmap2 = QPixmap('./ref/' + self.ComboBX.currentText() + '.jpg')
            self.label1.setPixmap(pixmap2)
            self.img_label1.setText("reference image")
            self.img_label2.setText("target image")
            save_ref_img(self.ComboBX.currentText())#
            self.warn_label.setText(" ")
            self.result_label1.setText(" ")
            self.result_label2.setText(" ")
            self.result_label3.setText(" ")

    def cmb_init(self):
        self.img_list = list(Path("./flood/").glob('*.jpg'))
        self.img_list = sorted(list(set([img.name for img in self.img_list])))
        self.length = len(self.img_list)

        self.img_list.insert(0," ")
        self.ComboBX = QComboBox(self)                      # image cmb
        self.ComboBX.move(20, 190)
        self.ComboBX.resize(100, 30)
        self.ComboBX.addItems(w[:-4] for w in self.img_list)
        self.ComboBX.insertSeparator(self.length + 1)
        self.ComboBX.currentIndexChanged.connect(self.cmb)

        self.ComboBX2 = QComboBox(self)                     # model cmb
        self.ComboBX2.move(20, 260)
        self.ComboBX2.resize(100, 30)
        self.ComboBX2.addItems([" ", "FCN", "DeepLabV3", "2DUnet", "Vnet", "3DUnet"])
        self.ComboBX2.insertSeparator(4)

    def img_init(self):

        self.img_label1 = QLabel(" ",self)
        self.img_label1.move(400, 390)
        self.img_label1.resize(200,20)
        self.img_label2 = QLabel(" ",self)
        self.img_label2.move(1000, 390)
        self.img_label2.resize(200,20)
        pixmap1 = QPixmap('./white_mask.png')
        self.label1.setPixmap(pixmap1)
        pixmap2 = QPixmap('./white_mask.png')
        self.label2.setPixmap(pixmap2)


def save_ref_img(img_name):
    weight = 0.7
    b = 1.0 - weight
    ref_img = np.asarray(cv2.imread("./ref/"+img_name+".jpg"))#
    gt = np.asarray(cv2.imread("./ref_gt/"+img_name+".png"))#
    height, width = gt.shape[:-1]#
    ref_tmp_img = ref_img#
    ref = np.zeros((height, width, 3))

    # create mask
    water_mask = (gt[:,:,0] == 255)
    road_mask = (gt[:,:,0] == 125)

    # make ref
    ref[road_mask] = np.array([0,255,0])
    ref[water_mask] = np.array([0,0,255])
    ref = ref*b + ref_img*weight
    ref = ref.astype(dtype=np.uint8)

    ref = Image.fromarray(ref)
    ref.save("./result/"+img_name+"_ref.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()