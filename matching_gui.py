import task3
from PyQt5 import QtWidgets , QtCore, QtGui
import matplotlib.pyplot as plt
from libs.harris import apply_harris
from libs.feature_matching import match
import time
import sys
import cv2
from libs.sift import SIFT
from PyQt5.QtWidgets import QMessageBox

class MainWindow(QtWidgets.QMainWindow , task3.Ui_MainWindow):
    # resized = QtCore.pyqtSignal()
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.widgets = [self.input_img,self.output_img,self.input_img_2,self.input_img_3, self.output_img_2]
        self.widget_configuration()
        self.default_img()
        self.open_button.clicked.connect(self.open_image)
        self.open_button_2.clicked.connect(self.open_image)
        self.apply_button.clicked.connect(self.harris_match)
        self.apply_button_2.clicked.connect(self.features_match)
        self.input_bgr = 2 * [0]
        self.input_rgb = 2 * [0]
    
    def open_image(self):
        if self.tabWidget.currentIndex( ) == 0:
            self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File',"", "Image Files (*.png *jpeg *.jpg)")
            self.img_bgr = cv2.imread(self.file_path)
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            
            self.apply_button.setEnabled(True)
            self.threshold.setEnabled(True)
            self.sensitivity.setEnabled(True)
            self.harris_input = self.img_bgr
            self.display(self.img_rgb,self.input_img)
            self.output_img.clear()
        else:
            self.file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open File',"", "Image Files (*.png *jpeg *.jpg)")
            if len(self.file_paths) == 2:
                for i in range(2):
                    self.input_bgr[i]=cv2.imread(self.file_paths[i])
                    self.input_rgb[i]=cv2.cvtColor(self.input_bgr[i], cv2.COLOR_BGR2RGB)
                    self.display(self.input_rgb[i],self.widgets[i+2])
                self.widgets[-1].clear()
                self.apply_button_2.setEnabled(True)
                self.radioButton.setEnabled(True)
                self.radioButton_2.setEnabled(True)
                self.threshold_3.setEnabled(True)
            else:
                self.pop_up()
                

    def harris_match(self):
        start = time.time()
        self.harris_output = apply_harris(self.harris_input,float(self.sensitivity.text()),float(self.threshold.text()))
        self.harris_output = cv2.cvtColor(self.harris_output, cv2.COLOR_BGR2RGB)
        end = time.time()
        self.display(self.harris_output,self.output_img)
        self.time_label.setText(str("{:.3f}".format(end-start)) + " Seconds")

    def features_match(self):
        if self.radioButton.isChecked():
            self.alg = "NCC"
        else:
            self.alg = "SSD"
        start = time.time()
        sf1 = SIFT(self.file_paths[0])
        kp1, descriptors_1 = sf1.computeKeypointsAndDescriptors()
        end = time.time()
        sift1_time = end - start
        start = time.time()
        sf2 = SIFT(self.file_paths[1])
        kp2, descriptors_2 = sf2.computeKeypointsAndDescriptors()
        end = time.time()
        sift2_time = end - start
        start = time.time()
        match_output = match(self.alg, float(self.threshold_3.text()),kp1, self.input_bgr[0],descriptors_1,self.input_bgr[1], kp2, descriptors_2)
        end = time.time()
        match_time = end - start
        total_time = match_time + sift2_time + sift1_time
        self.time_label_2.setText(str("{:.3f}".format(sift1_time)) + " Seconds")
        self.time_label_5.setText(str("{:.3f}".format(sift2_time)) + " Seconds")
        self.time_label_4.setText(str("{:.3f}".format(match_time)) + " Seconds")
        self.time_label_6.setText(str("{:.3f}".format(total_time)) + " Seconds")
        match_output = cv2.cvtColor(match_output, cv2.COLOR_BGR2RGB)
        self.display(match_output,self.widgets[-1])
        
        

    def display(self , data , widget):
            data = cv2.transpose(data)
            widget.setImage(data)
            widget.view.setLimits(xMin=0, xMax=data.shape[0], yMin= 0 , yMax= data.shape[1])
            widget.view.setRange(xRange=[0, data.shape[0]], yRange=[0, data.shape[1]], padding=0)    

    def widget_configuration(self):

        for widget in self.widgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def default_img(self):
        defaultImg = plt.imread("images/default-image.jpg")
        for widget in self.widgets:
            self.display(defaultImg,widget)
    def pop_up(self):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText('Warning!')
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setInformativeText('You must select only 2 images!')
        x = msg.exec_()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()