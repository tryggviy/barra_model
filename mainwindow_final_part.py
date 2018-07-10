# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow5.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from jxhy_general_final import main
from PyQt5.QtCore import pyqtSignal, QThread, QRect, QDate, QDateTime, QMetaObject, QCoreApplication, QTime
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QTextEdit, QLabel, QPushButton, QTextBrowser, QDateEdit, QSpinBox, QMenuBar, QMenu, QStatusBar, QFileDialog, QApplication, QMainWindow
from time import sleep


def changeTime(alltime):
    hour = 60 * 60
    min = 60
    hournum = (alltime) // hour
    minnum = (alltime - hournum*hour)//60
    secnum = alltime - hournum*hour - minnum*min
    return ("%d hours, %d minutes, %d seconds" %(hournum, minnum, secnum))


class MyThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self, parent=None):
        super(MyThread, self).__init__(parent)

    def run(self):
        count = 0
        while True:
            self.trigger.emit("\n____________Barra_Analysis_____________"
                              "\nPlease wait, calculating...\n"+changeTime(count)+" passed")  #发射信号
            count += 10
            sleep(10)

    def stop_(self):
        self.isRunning = False
        print('ending...')
        self.terminate()

class MyMainThread(QThread):
    result = pyqtSignal(str)

    def __init__(self, parent=None):
        super(MyMainThread, self).__init__(parent)
        self.parameters_in = []
        self.str_result = []

    def run(self):
        sleep(1)
        print(self.parameters_in)
        
        # self.str_result = main(self.parameters_in[0],self.parameters_in[1],self.parameters_in[2],\
        #     self.parameters_in[3],self.parameters_in[4],self.parameters_in[5])
        # self.result.emit('Calculation Finished...')
        try:
            self.str_result = main(self.parameters_in[0],self.parameters_in[1],self.parameters_in[2],\
                self.parameters_in[3],self.parameters_in[4],self.parameters_in[5])
        except:
            self.result.emit('\nThe calculation process has encountered fatal error\n')
        else:
            self.result.emit('\nCalculation Finished...\n')
            
    def stop_(self):
        self.isRunning = False
        print('ending...')
        self.terminate()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textSuffix = QTextEdit(self.centralwidget)
        self.textSuffix.setGeometry(QRect(20, 230, 261, 31))
        self.textSuffix.setObjectName("textSuffix")
        self.textRiskf = QTextEdit(self.centralwidget)
        self.textRiskf.setGeometry(QRect(20, 360, 261, 31))
        self.textRiskf.setObjectName("textRiskf")
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QRect(20, 330, 261, 20))
        self.label.setObjectName("label")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setGeometry(QRect(20, 270, 281, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setGeometry(QRect(20, 200, 261, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setGeometry(QRect(20, 130, 261, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setGeometry(QRect(20, 60, 261, 21))
        self.label_5.setObjectName("label_5")
        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setGeometry(QRect(20, 510, 111, 31))
        self.startButton.setObjectName("startButton")
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setGeometry(QRect(20, 10, 261, 41))
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.resultBrowser = QTextBrowser(self.centralwidget)
        self.resultBrowser.setGeometry(QRect(310, 30, 461, 501))
        self.resultBrowser.setObjectName("resultBrowser")
        self.DateStart = QDateEdit(self.centralwidget)
        self.DateStart.setGeometry(QRect(20, 90, 121, 31))
        self.DateStart.setMaximumDate(QDate(2050, 12, 31))
        self.DateStart.setMinimumDate(QDate(1990, 12, 31))
        self.DateStart.setDate(QDate(2018, 1, 1))
        self.DateStart.setObjectName("DateStart")
        self.DateEnd = QDateEdit(self.centralwidget)
        self.DateEnd.setGeometry(QRect(19, 163, 121, 31))
        self.DateEnd.setDateTime(QDateTime(QDate(2018, 1, 1), QTime(0, 0, 0)))
        self.DateEnd.setMaximumDateTime(QDateTime(QDate(2050, 12, 31), QTime(23, 59, 59)))
        self.DateEnd.setMinimumDate(QDate(1990, 12, 31))
        self.DateEnd.setObjectName("DateEnd")
        self.spinBoxMode = QSpinBox(self.centralwidget)
        self.spinBoxMode.setGeometry(QRect(20, 290, 71, 31))
        self.spinBoxMode.setMinimum(1)
        self.spinBoxMode.setMaximum(3)
        self.spinBoxMode.setObjectName("spinBoxMode")
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setGeometry(QRect(170, 300, 101, 21))
        self.label_7.setObjectName("label_7")
        self.endButton = QPushButton(self.centralwidget)
        self.endButton.setGeometry(QRect(160, 510, 121, 31))
        self.endButton.setObjectName("endButton")
        self.fileButton = QPushButton(self.centralwidget)
        self.fileButton.setGeometry(QRect(20, 410, 261, 31))
        self.fileButton.setObjectName("fileButton")
        self.pathBrowser = QTextBrowser(self.centralwidget)
        self.pathBrowser.setGeometry(QRect(20, 450, 261, 41))
        self.pathBrowser.setObjectName("pathBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menuAnalysis = QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuAnalysis.menuAction())

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)
        ####################################################

        self.mythread = MyThread()
        self.mainthread = MyMainThread()

        self.fileButton.clicked.connect(lambda : self.file_path())
        self.mythread.trigger.connect(self.update_text)
        self.mainthread.result.connect(self.update_result)

        self.startButton.clicked.connect(lambda : self.input_Parameters())
        self.startButton.clicked.connect(lambda : self.mainthread.start())
        self.startButton.clicked.connect(lambda : self.mythread.start())
        self.endButton.clicked.connect(lambda : self.end_calculation())

        
    def input_Parameters(self):
        self.aa = str(self.DateStart.date().toString("yyyyMMdd"))
        self.bb = str(self.DateEnd.date().toString("yyyyMMdd"))
        self.cc = str(self.textSuffix.toPlainText())
        self.dd = int(self.spinBoxMode.value())
        self.ee = float(self.textRiskf.toPlainText())

        if self.dd==1:
            self.dx='p1f1'
        elif self.dd==2:
            self.dx='p0f1'
        elif self.dd==3:
            self.dx='p1f0'
        else:
            raise Exception('Running Mode is wrong')

        self.mainthread.parameters_in = [self.aa ,self.bb, self.cc, self.dx, self.ee, self.directory1]


    def file_path(self):  
        self.directory1 = QFileDialog.getExistingDirectory(self.centralwidget,"Please choose folder","/")+'/'
        self.pathBrowser.append(str(self.directory1))

    def update_text(self, message):
        self.resultBrowser.append(str(message))
    
    def update_result(self,message):
        self.mythread.stop_()
        sleep(1)
        self.resultBrowser.append(str(message))
        print(self.mainthread.str_result)
        for i in self.mainthread.str_result:
            print(i)
            self.resultBrowser.append(i)

    def end_calculation(self):
        self.mythread.stop_()
        self.mainthread.stop_()
        sleep(1)
        self.resultBrowser.append('\nCalculation terminated by user...')


    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Please input risk free rate(0.035)"))
        self.label_2.setText(_translate("MainWindow", "Choose a running mode(e.g (1)p1f1/(2)p0f1"))
        self.label_3.setText(_translate("MainWindow", "Please input the suffix for result file"))
        self.label_4.setText(_translate("MainWindow", "Please input earliest date(e.g 2018-01-01)"))
        self.label_5.setText(_translate("MainWindow", "Please input latest date(e.g 2018-12-30)"))
        self.startButton.setText(_translate("MainWindow", "Start Analysis"))
        self.label_6.setText(_translate("MainWindow", "Barra Contribution Analysis(v_test)"))
        self.label_7.setText(_translate("MainWindow", " (3)p1f0"))
        self.endButton.setText(_translate("MainWindow", "End Process"))
        self.fileButton.setText(_translate("MainWindow", "Choose a folder"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())